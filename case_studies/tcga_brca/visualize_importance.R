###############################################################################
## Visualize the global importances from sweep.py. Settings are available in
## config.yaml.
###############################################################################

library(tidyverse)
library(tidytext)
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

pca_normalize_mass <- cfg$visualization$pca_normalize_mass
if (is.null(pca_normalize_mass)) pca_normalize_mass <- TRUE

methods <- names(keep(cfg$methods, isTRUE))
categories <- tibble(method = names(cfg$categories), category = cfg$categories) |>
    unnest(category)

theme_set(theme_bw(base_size = 10))

# --- Helpers -----------------------------------------------------------------
# strip_prefix, parse_omic, omic_colors, sweep_sample_size,
# has_nonzero_variance, load_importance_data, and importance_pca live in
# importance_visualize_helpers.R since they're shared with pc1_diagnostics.R.

# PCA of method importance profiles
pca_plot <- function(mat, normalize_mass = TRUE) {
    pca <- importance_pca(mat, normalize_mass)
    pca_df <- as_tibble(pca$x[, 1:2], rownames = "method")
    ve <- summary(pca)$importance[2, 1:2] * 100

    ggplot(pca_df, aes(PC1, PC2, label = method)) +
        geom_point(size = 3, color = "#767575") +
        geom_text_repel(size = 3, max.overlaps = 20) +
        labs(
            title = "(a) Method scores",
            x = glue("PC1 ({round(ve[1], 1)}%)"),
            y = glue("PC2 ({round(ve[2], 1)}%)")
        )
}

# Largest feature loadings for the first two principal components
pca_loadings_plot <- function(
    mat, n_features = 50, normalize_mass = TRUE
) {
    pca <- importance_pca(mat, normalize_mass)
    ve <- summary(pca)$importance[2, 1:2] * 100
    dim_labels <- c(
        PC1 = glue("PC1 ({round(ve[1], 1)}%)"),
        PC2 = glue("PC2 ({round(ve[2], 1)}%)")
    )

    loadings <- as_tibble(
        pca$rotation[, 1:2, drop = FALSE], rownames = "feature"
    ) |>
        pivot_longer(c(PC1, PC2), names_to = "dim", values_to = "loading") |>
        group_by(dim) |>
        slice_max(abs(loading), n = n_features, with_ties = FALSE) |>
        ungroup() |>
        mutate(
            dim = recode(dim, !!!dim_labels),
            omic = parse_omic(feature),
            feature_ordered = reorder_within(
                str_to_lower(strip_prefix(feature)), loading, dim
            ),
            sign = if_else(loading >= 0, "positive", "negative")
        )

    ggplot(loadings, aes(feature_ordered, omic)) +
        geom_point(
            aes(size = abs(loading), fill = sign),
            shape = 22, color = "black", stroke = 0.4
        ) +
        facet_wrap(~dim, ncol = 1, scales = "free_x") +
        scale_x_reordered() +
        scale_size(range = c(2, 12), name = "|Loading|") +
        scale_fill_manual(
            values = c(positive = "black", negative = "white"),
            name = "Sign"
        ) +
        labs(
            title = "(b) Feature loadings",
            x = "Feature", y = "Measurement Assay"
        ) +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
            panel.background = element_rect(fill = "#cecdcd", color = NA),
            panel.grid.major = element_blank(),
            strip.text = element_text(size = 10)
        )
}

# PCA scores above two side-by-side panels of their associated loadings
pca_figure <- function(mat, n_features = 100, normalize_mass = TRUE) {
    scores <- pca_plot(mat, normalize_mass)
    loadings <- pca_loadings_plot(mat, n_features, normalize_mass)

    ((scores / loadings) +
        plot_layout(heights = c(1, 1), guides = "collect")) &
        theme(legend.position = "bottom")
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
        geom_text_repel(aes(label = label), size = 5, max.overlaps = 20, show.legend = FALSE) +
        scale_color_manual(values = omic_colors, name = "Omic") +
        labs(
            x = "SAGE",
            y = "minSHAP",
            title = "SAGE vs minSHAP feature importance"
        )
}

# Normalize each method's absolute importance over all retained features
normalize_importance_mass <- function(importance, method_cols) {
    importance |>
        pivot_longer(-feature, names_to = "method", values_to = "score") |>
        filter(method %in% method_cols) |>
        group_by(method) |>
        mutate(total_mass = sum(abs(score), na.rm = TRUE)) |>
        ungroup() |>
        transmute(
            feature, method,
            mass = if_else(total_mass > 0, abs(score) / total_mass, NA_real_)
        )
}

# ICE plot for the TCGA example with predictions overlaid
vignette_line_panel <- function(
    feat, vignette_pdp, vignette_ice, vignette_predictions, class_labels,
    show_y_title = FALSE
) {
    pdp_sub <- vignette_pdp |> filter(feature == feat)
    ice_sub <- vignette_ice |> filter(feature == feat)
    prediction_sub <- vignette_predictions |>
        filter(feature == feat) |>
        mutate(
            actual_class = factor(
                actual_class, levels = 0:1, labels = class_labels
            )
        )

    ggplot() +
        geom_line(
            data = ice_sub,
            aes(grid_value, ice_value, group = sample_index),
            color = "grey40", alpha = 0.10, linewidth = 0.15
        ) +
        geom_line(
            data = pdp_sub,
            aes(grid_value, pdp_value),
            color = "black", linewidth = 0.8
        ) +
        geom_point(
            data = prediction_sub,
            aes(feature_value, prediction, color = actual_class),
            alpha = 0.55, size = 0.8
        ) +
        geom_jitter(
            data = prediction_sub,
            aes(feature_value, predicted_class, color = actual_class),
            width = 0, height = 0.025, alpha = 0.45, size = 0.8
        ) +
        scale_color_manual(
            values = c("#0072B2", "#D55E00"), name = "Actual class",
            drop = FALSE,
            guide = guide_legend(override.aes = list(size = 4))
        ) +
        scale_y_continuous(
            limits = c(-0.05, 1.05), breaks = c(0, 0.5, 1),
            name = if (show_y_title) "Prediction probability" else NULL
        ) +
        labs(x = "Feature value", title = strip_prefix(feat)) +
        theme(plot.title = element_text(hjust = 0.5))
}

vignette_bar_panel <- function(feat, importance_mass) {
    feat_scores <- importance_mass |>
        filter(feature == feat) |>
        mutate(method = fct_reorder(method, mass))

    ggplot(feat_scores, aes(mass, method)) +
        geom_col() +
        scale_x_continuous(labels = scales::label_percent(accuracy = 0.1)) +
        labs(x = "Fraction of absolute importances", y = NULL) +
        theme(axis.text.y = element_text(size = 12))
}

# Combine the per-gene ice plots into the final case study gene-level figure.
vignette_plot <- function(
    importance, mat, method_cols, k = 3, normalize_mass = TRUE
) {
    importance_mass <- normalize_importance_mass(importance, method_cols)
    top_feats <- importance_pca(mat, normalize_mass)$rotation[, "PC2"] |>
        enframe(name = "feature", value = "loading") |>
        slice_max(abs(loading), n = k, with_ties = FALSE) |>
        pull(feature)

    vp <- read_csv(path(res, "vignette_pdp.csv"), show_col_types = FALSE) |>
        filter(feature %in% top_feats)
    ice <- read_csv(path(res, "vignette_ice.csv"), show_col_types = FALSE) |>
        filter(feature %in% top_feats)
    predictions <- read_csv(
        path(res, "vignette_predictions.csv"), show_col_types = FALSE
    ) |>
        filter(feature %in% top_feats)
    class_labels <- cfg$outcome$classes

    line_panels <- map2(
        top_feats, seq_along(top_feats),
        \(f, i) vignette_line_panel(
            f, vp, ice, predictions, class_labels, show_y_title = i == 1
        )
    )
    bar_panels <- map(top_feats, \(f) vignette_bar_panel(f, importance_mass))

    wrap_plots(c(line_panels, bar_panels), ncol = k) +
        plot_layout(guides = "collect") &
        theme(
            legend.position = "bottom",
            plot.title = element_text(size = 14, face = "bold"),
            axis.title = element_text(size = 11),
            legend.title = element_text(size = 12),
            legend.text = element_text(size = 11)
        )
}

# --- Run ---------------------------------------------------------------------

# Load & prepare data
n <- sweep_sample_size(res, cfg$dataset, cfg$response_type)
d <- load_importance_data(cfg$dataset, n, cfg$response_type, cfg$seeds, methods, res)

# Figure 1 — PCA scores and feature loadings
ggsave(
    path(res, "fig_method_pca.pdf"),
    pca_figure(d$mat, normalize_mass = pca_normalize_mass),
    width = 8, height = 8
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
    vignette_plot(
        d$importance, d$mat, d$method_cols,
        normalize_mass = pca_normalize_mass
    ),
    width = 12, height = 7
)

# Figure 4 — SAGE vs minSHAP credit-splitting scatter
ggsave(
    path(res, "fig_credit_splitting.pdf"),
    credit_splitting_plot(d$importance),
    width = 8, height = 6
)
