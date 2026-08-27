###############################################################################
## Visualize the global importances from sweep.py. Settings are available in
## config.yaml.
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

# --- Paths & config ----------------------------------------------------------

base <- here("case_studies", "tcga_brca")
res <- path(base, "results")
cfg <- yaml::read_yaml(path(base, "config.yaml"))

methods <- names(keep(cfg$methods, isTRUE))
categories <- tibble(method = names(cfg$categories), category = cfg$categories) |>
    unnest(category)

theme_set(theme_bw(base_size = 10))

# --- Helpers -----------------------------------------------------------------

strip_prefix <- \(x) str_remove(x, "^(mrna|mirna|protein)__")
parse_omic <- \(x) str_extract(x, "^[^_]+")
omic_colors <- c(mrna = "#2b8cbe", mirna = "#e34a33", protein = "#31a354")

#' Sample size used in the result filenames, read back from the sweep's output.
sweep_sample_size <- function() {
    stems <- path_file(dir_ls(res, glob = glue("*_{cfg$response_type}_*.csv")))
    sizes <- str_match(stems, glue("^{cfg$dataset}_([0-9]+)_{cfg$response_type}_"))[, 2]
    unique(na.omit(sizes)) |> as.numeric()
}

#' Average each method's importance profile over seeds.
#'
#' Returns the same shape `pca_plot()` and `corr_plot()` expect: features in
#' rows, methods in columns, all-NA columns dropped.
load_importance_data <- function(n) {
    mats <- map(
        cfg$seeds,
        \(seed) importance_matrix(
            cfg$dataset, n, cfg$response_type, seed, methods, res
        )
    )
    features <- rownames(mats[[1]])
    averaged <- reduce(map(mats, as.matrix), `+`) / length(mats)

    keep_cols <- colSums(!is.na(averaged)) > 0
    imp <- as_tibble(averaged, rownames = "feature")
    list(
        importance = imp,
        mat = averaged[, keep_cols, drop = FALSE],
        method_cols = colnames(averaged)[keep_cols],
        features = features
    )
}

# PCA of methods colored by null-importance category
pca_plot <- function(mat) {
    mat_z <- scale(mat)
    mat_z[is.na(mat_z)] <- 0
    pca <- prcomp(t(mat_z), center = FALSE, scale. = FALSE)
    pca_df <- as_tibble(pca$x[, 1:2], rownames = "method") |>
        left_join(categories, by = "method")

    ve <- summary(pca)$importance[2, 1:2] * 100

    ggplot(pca_df, aes(PC1, PC2, label = method)) +
        geom_point(size = 3) +
        geom_text_repel(size = 3, max.overlaps = 20) +
        labs(
            title = "PCA of importance methods",
            x = glue("PC1 ({round(ve[1], 1)}%)"),
            y = glue("PC2 ({round(ve[2], 1)}%)")
        )
}

# Spearman correlation heatmap with hierarchical clustering
corr_plot <- function(mat) {
    cor_mat <- cor(mat, method = "spearman", use = "pairwise.complete.obs")
    hc <- hclust(as.dist(1 - cor_mat), method = "ward.D2")
    order <- hc$labels[hc$order]

    cor_df <- cor_mat |>
        as_tibble(rownames = "method1") |>
        pivot_longer(-method1, names_to = "method2", values_to = "rho") |>
        mutate(
            method1 = factor(method1, levels = order),
            method2 = factor(method2, levels = order)
        )

    ggplot(cor_df, aes(method1, method2, fill = rho)) +
        geom_tile() +
        geom_text(aes(label = round(rho, 2)), size = 2.5) +
        scale_x_discrete(expand = c(0, 0)) +
        scale_y_discrete(expand = c(0, 0)) +
        scale_fill_scico(palette = "glasgow", limits = c(-1, 1), name = "Spearman\nrho") +
        labs(x = NULL, y = NULL, title = "Rank correlation between importance methods") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot mean SAGE vs. mean minSHAP per feature, colored by omic block
credit_splitting_plot <- function(importance) {
    feat_summary <- importance |>
        transmute(
            feature,
            sage = sage,
            minshap = minshap,
            omic = parse_omic(feature)
        ) |>
        mutate(
            residual = abs(minshap - sage),
            label = if_else(
                residual > quantile(residual, 0.99) | sage > quantile(sage, 0.99),
                strip_prefix(feature),
                NA_character_
            )
        )

    ggplot(feat_summary, aes(sage, minshap, color = omic)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
        geom_point(size = 0.5) +
        geom_text_repel(aes(label = label), size = 3, max.overlaps = 20, show.legend = FALSE) +
        scale_color_manual(values = omic_colors, name = "Omic") +
        labs(
            x = "SAGE",
            y = "minSHAP",
            title = "SAGE vs minSHAP feature importance"
        )
}

# Build one vignette panel: importance bar chart + PDP line plot
vignette_panel <- function(feat, importance, method_cols, vignette_pdp) {
    feat_scores <- importance |>
        filter(feature == feat) |>
        pivot_longer(-feature, names_to = "method", values_to = "score") |>
        filter(method %in% method_cols) |>
        left_join(categories, by = "method") |>
        mutate(method = fct_reorder(method, score))

    p_bar <- ggplot(feat_scores, aes(score, method)) +
        geom_col() +
        labs(x = "Importance", y = NULL, title = strip_prefix(feat))

    pdp_sub <- vignette_pdp |> filter(feature == feat)
    p_pdp <- ggplot(pdp_sub, aes(grid_value, pdp_value)) +
        geom_line(linewidth = 1) +
        labs(x = strip_prefix(feat), y = "Partial dependence")

    p_bar | p_pdp
}

# Full vignette figure: top-k features as stacked panels
vignette_plot <- function(importance, method_cols, k = 3) {
    vp <- read_csv(path(res, "vignette_pdp.csv"), show_col_types = FALSE)
    top_feats <- head(unique(vp$feature), k)

    panels <- map(top_feats, \(f) vignette_panel(f, importance, method_cols, vp))
    wrap_plots(panels, ncol = 1)
}

# --- Run ---------------------------------------------------------------------

# Load & prepare data
d <- load_importance_data(sweep_sample_size())

# Figure 1 — PCA bi-plot
ggsave(
    path(res, "fig_method_pca.pdf"),
    pca_plot(d$mat),
    width = 8, height = 6
)

# Figure 2 — Spearman correlation heatmap
ggsave(
    path(res, "fig_method_corr.pdf"),
    corr_plot(d$mat),
    width = 8, height = 7
)

# Figure 3 — Vignette panels
ggsave(
    path(res, "fig_vignettes.pdf"),
    vignette_plot(d$importance, d$method_cols),
    width = 10, height = 12
)

# Figure 4 — SAGE vs minSHAP credit-splitting scatter
ggsave(
    path(res, "fig_credit_splitting.pdf"),
    credit_splitting_plot(d$importance),
    width = 8, height = 6
)
