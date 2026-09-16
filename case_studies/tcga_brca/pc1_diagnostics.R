###############################################################################
## Diagnostics for the method PCA in visualize_importance.R.
##
## PC1: checks three candidate explanations for why PC1's largest loadings
## are uniformly signed: (1) methods that concentrate importance mass on
## fewer features, (2) features that are more correlated with the rest of
## the raw data, and (3) the marginal/conditional/risk/functional
## null-importance categories already recorded in config.yaml (see
## ../Null_Importance.pdf, sec 8.3, on redundancy as a source of divergence
## between these notions).
##
## PC2: checks whether it separates methods by null-importance category (it
## does: conditional-notion methods score positive, functional-notion
## methods score negative) and whether its feature loadings track how
## concentrated each feature's PDP curve is (a sharp local jump vs. a
## gradual change spread across the range) -- an empirical instance of
## ../Null_Importance.pdf's global/local/pointwise functional null
## importance distinction (Definitions 2.2-2.4), which frames these as
## different spatial scales of functional relevance.
##
## Run after visualize_importance.R, feature_correlation.py, and
## pdp_concentration.py.
###############################################################################

library(tidyverse)
library(scico)
library(ggrepel)
library(patchwork)
library(yaml)
library(here)
library(fs)
library(glue)

source(here("case_studies", "src", "R", "importance_visualize_helpers.R"))

# --- Paths & config ------------------------------------------------------

base <- here("case_studies", "tcga_brca")
res <- path(base, "results")
cfg <- yaml::read_yaml(path(base, "config.yaml"))

pca_normalize_mass <- cfg$visualization$pca_normalize_mass
if (is.null(pca_normalize_mass)) pca_normalize_mass <- TRUE

methods <- names(keep(cfg$methods, isTRUE))
categories <- tibble(method = names(cfg$categories), category = cfg$categories) |>
    unnest(category)

theme_set(theme_bw(base_size = 14))

# --- Helpers ---------------------------------------------------------------

# Each method's importance profile as a fraction of its total absolute mass;
# mirrors importance_pca()'s normalize_mass = TRUE step, exposed here since
# the diagnostics below need the matrix, not just the PCA fit.
mass_normalize <- function(mat) {
    total_mass <- colSums(abs(mat), na.rm = TRUE)
    sweep(abs(mat), 2, total_mass, `/`)
}

#' Gini coefficient of a nonnegative vector (0 = perfectly even, 1 = all mass
#' on one entry).
gini_coefficient <- function(x) {
    x <- sort(x[is.finite(x) & x >= 0])
    n <- length(x)
    if (n == 0 || sum(x) == 0) {
        return(NA_real_)
    }
    (2 * sum(seq_len(n) * x)) / (n * sum(x)) - (n + 1) / n
}

#' Fraction of total mass held by the k largest entries of a nonnegative
#' vector.
top_k_mass_fraction <- function(x, k) {
    x <- x[is.finite(x) & x >= 0]
    if (sum(x) == 0) {
        return(NA_real_)
    }
    sum(sort(x, decreasing = TRUE)[seq_len(min(k, length(x)))]) / sum(x)
}

#' Per-method concentration metrics, computed directly from the
#' mass-normalized profile (independent of the PCA).
concentration_metrics <- function(mat, top_k = 20) {
    profiles <- mass_normalize(mat)
    tibble(
        method = colnames(profiles),
        gini = apply(profiles, 2, gini_coefficient),
        top_k_frac = apply(profiles, 2, top_k_mass_fraction, k = top_k)
    )
}

# PC1/PC2 method scores by null-importance category
category_score_plot <- function(pca, categories) {
    scores <- as_tibble(pca$x[, 1:2], rownames = "method") |>
        inner_join(categories, by = "method") |>
        pivot_longer(c(PC1, PC2), names_to = "dim", values_to = "score")

    ggplot(scores, aes(score, category, color = category)) +
        geom_vline(xintercept = 0, color = "grey70") +
        geom_point(size = 2.5) +
        geom_text_repel(
            aes(label = method), size = 2.8, max.overlaps = 20, show.legend = FALSE
        ) +
        facet_wrap(~dim, ncol = 1, scales = "free_x") +
        scale_color_scico_d(palette = "berlin", guide = "none") +
        labs(
            title = "Method score by null-importance category",
            x = "Score", y = "Category"
        )
}

# Each feature's PC1 loading against its correlation with the rest of the
# raw data (feature_correlation.csv, from feature_correlation.py)
loading_correlation_plot <- function(pca, feature_correlation) {
    loadings <- as_tibble(pca$rotation[, "PC1", drop = FALSE], rownames = "feature") |>
        inner_join(feature_correlation, by = "feature") |>
        mutate(omic = parse_omic(feature))
    rho <- cor(loadings$PC1, loadings$mean_abs_corr, method = "spearman", use = "complete.obs")

    ggplot(loadings, aes(mean_abs_corr, PC1, color = omic)) +
        geom_point(size = 1.2, alpha = 0.7) +
        scale_color_manual(values = omic_colors, name = "Omic") +
        labs(
            title = "PC1 loading vs. feature's correlation with the rest of the data",
            subtitle = glue("Spearman rho = {round(rho, 2)}"),
            x = "Mean |correlation| with other features",
            y = "PC1 loading"
        )
}

# Each feature's PC2 loading against the concentration of its PDP curve's
# step-to-step change (pdp_concentration.csv, from pdp_concentration.py):
# a sharp, localized jump (high pdp_gini) vs. a gradual change spread across
# the whole range (low pdp_gini)
pdp_concentration_plot <- function(pca, pdp_concentration) {
    loadings <- as_tibble(pca$rotation[, "PC2", drop = FALSE], rownames = "feature") |>
        inner_join(pdp_concentration, by = "feature") |>
        mutate(omic = parse_omic(feature))
    rho <- cor(loadings$PC2, loadings$pdp_gini, method = "spearman", use = "complete.obs")

    ggplot(loadings, aes(pdp_gini, PC2, color = omic)) +
        geom_point(size = 1.2, alpha = 0.7) +
        scale_color_manual(values = omic_colors, name = "Omic") +
        labs(
            title = "PC2 loading vs. concentration of the feature's PDP curve",
            subtitle = glue("Spearman rho = {round(rho, 2)}"),
            x = "PDP curve Gini (sharp local jump vs. gradual change)",
            y = "PC2 loading"
        )
}

# Each method's PC1 score against its importance-mass concentration
concentration_scatter_plot <- function(pca, concentration) {
    scores <- as_tibble(pca$x[, "PC1", drop = FALSE], rownames = "method") |>
        inner_join(concentration, by = "method") |>
        pivot_longer(c(gini, top_k_frac), names_to = "metric", values_to = "value")

    rhos <- scores |>
        group_by(metric) |>
        summarize(rho = cor(PC1, value, method = "spearman"), .groups = "drop") |>
        mutate(label = glue("rho = {round(rho, 2)}"))

    ggplot(scores, aes(value, PC1, label = method)) +
        geom_point(size = 2, color = "#767575") +
        geom_text_repel(size = 2.8, max.overlaps = 20) +
        geom_text(
            data = rhos, aes(x = -Inf, y = Inf, label = label),
            hjust = -0.1, vjust = 1.5, inherit.aes = FALSE, size = 3
        ) +
        facet_wrap(~metric, scales = "free_x") +
        labs(
            title = "Method PC1 score vs. importance-mass concentration",
            x = "Concentration metric", y = "PC1 score"
        )
}

# Cumulative importance-mass vs. feature-rank curves for the methods with the
# most extreme PC1 scores -- a PCA-independent visual check of concentration
concentration_curve_plot <- function(mat, pca, n_show = 11) {
    profiles <- mass_normalize(mat)
    pc1 <- pca$x[, "PC1"]
    extremes <- c(
        names(sort(pc1))[seq_len(n_show)],
        names(sort(pc1, decreasing = TRUE))[seq_len(n_show)]
    ) |>
    unique()

    curves <- map_dfr(extremes, \(m) {
        sorted <- sort(profiles[, m], decreasing = TRUE)
        tibble(
            method = m,
            rank = seq_along(sorted),
            rank_frac = seq_along(sorted) / length(sorted),
            cum_mass = cumsum(sorted)
        )
    }) |>
    mutate(method = factor(method, levels = extremes))

    ggplot(curves, aes(rank, cum_mass, color = method)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey70") +
        geom_line(linewidth = 0.8) +
        scale_color_scico_d(palette = "berlin", name = "Method") +
        labs(
            x = "Feature rank",
            y = "Cumulative importance"
        )
}

# --- Run ---------------------------------------------------------------------

n <- sweep_sample_size(res, cfg$dataset, cfg$response_type)
d <- load_importance_data(cfg$dataset, n, cfg$response_type, cfg$seeds, methods, res)
pca <- importance_pca(d$mat, normalize_mass = pca_normalize_mass)

feature_correlation <- read_csv(path(res, "feature_correlation.csv"), show_col_types = FALSE)
pdp_concentration <- read_csv(path(res, "pdp_concentration.csv"), show_col_types = FALSE)
concentration <- concentration_metrics(d$mat)

# Consensus-alignment control: for uncentered PCA of nonnegative data, PC1's
# loading vector is expected to closely track the across-method mean profile.
profiles <- mass_normalize(d$mat)
consensus_rho <- cor(pca$rotation[, "PC1"], rowMeans(profiles), method = "spearman")
message(glue("PC1 loading vs. across-method mean profile: rho = {round(consensus_rho, 3)}"))

category_summary <- as_tibble(pca$x[, 1:2], rownames = "method") |>
    inner_join(categories, by = "method") |>
    group_by(category) |>
    summarize(mean_pc1 = mean(PC1), mean_pc2 = mean(PC2), .groups = "drop") |>
    arrange(mean_pc1)
message("Mean PC1/PC2 score by null-importance category:")
print(category_summary)

panels <- (category_score_plot(pca, categories) /
    loading_correlation_plot(pca, feature_correlation) /
    pdp_concentration_plot(pca, pdp_concentration) /
    concentration_scatter_plot(pca, concentration) /
    concentration_curve_plot(d$mat, pca)) +
    plot_annotation(title = "PC1/PC2 diagnostics") &
    theme(legend.position = "bottom")

ggsave(path(res, "fig_pc1_diagnostics.pdf"), panels, width = 7, height = 19)

summary_table <- as_tibble(pca$x[, 1:2], rownames = "method") |>
    inner_join(concentration, by = "method") |>
    left_join(
        categories |>
            group_by(method) |>
            summarize(categories = paste(category, collapse = ", "), .groups = "drop"),
        by = "method"
    )
write_csv(summary_table, path(res, "pc1_diagnostics.csv"))
