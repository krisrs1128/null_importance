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

strip_prefix <- \(x) str_remove(x, "^(mrna|mirna|protein)__")
parse_omic <- \(x) str_extract(x, "^[^_]+")
omic_colors <- c(mrna = "#2b8cbe", mirna = "#e34a33", protein = "#31a354")

#' Sample size used in the result filenames, read back from the sweep's output.
sweep_sample_size <- function() {
    stems <- path_file(dir_ls(res, glob = glue("*_{cfg$response_type}_*.csv")))
    sizes <- str_match(stems, glue("^{cfg$dataset}_([0-9]+)_{cfg$response_type}_"))[, 2]
    unique(na.omit(sizes)) |> as.numeric()
}

#' Whether a vector has an estimable, nonzero variance.
has_nonzero_variance <- function(x) {
    x <- x[is.finite(x)]
    length(x) > 1 && var(x) > 0
}

#' Average each method's importance profile over seeds.
#'
#' Returns the same shape `pca_plot()` and `corr_plot()` expect: features in
#' rows and methods in columns. Constant features and method profiles are
#' removed because they cannot be standardized or correlated.
load_importance_data <- function(n) {
    mats <- map(
        cfg$seeds,
        \(seed) importance_matrix(
            cfg$dataset, n, cfg$response_type, seed, methods, res
        )
    )
    features <- rownames(mats[[1]])
    averaged <- reduce(map(mats, as.matrix), `+`) / length(mats)

    keep_cols <- apply(averaged, 2, has_nonzero_variance)
    mat <- averaged[, keep_cols, drop = FALSE]
    keep_features <- apply(mat, 1, has_nonzero_variance)

    imp <- as_tibble(averaged, rownames = "feature") |>
        filter(keep_features)
    list(
        importance = imp,
        mat = mat[keep_features, , drop = FALSE],
        method_cols = colnames(mat),
        features = features[keep_features]
    )
}

# Fit PCA with methods as observations and features as variables
importance_pca <- function(mat, normalize_mass = TRUE) {
    if (normalize_mass) {
        total_mass <- colSums(abs(mat), na.rm = TRUE)
        mat_pca <- sweep(abs(mat), 2, total_mass, `/`)
    } else {
        mat_pca <- scale(mat)
    }
    mat_pca[!is.finite(mat_pca)] <- 0

    pca_input <- t(mat_pca)
    attr(pca_input, "scaled:center") <- NULL
    attr(pca_input, "scaled:scale") <- NULL
    prcomp(pca_input, center = FALSE, scale. = FALSE)
}

# PCA of method importance profiles
pca_plot <- function(mat, normalize_mass = TRUE) {
    pca <- importance_pca(mat, normalize_mass)
    pca_df <- as_tibble(pca$x[, 1:2], rownames = "method")
    ve <- summary(pca)$importance[2, 1:2] * 100

    ggplot(pca_df, aes(PC1, PC2, label = method)) +
        geom_point(size = 3) +
        geom_text_repel(size = 3, max.overlaps = 20) +
        labs(
            title = "PCA of TCGA Importances",
            x = glue("PC1 ({round(ve[1], 1)}%)"),
            y = glue("PC2 ({round(ve[2], 1)}%)")
        )
}

# Largest feature loadings for the first two principal components
pca_loadings_plot <- function(
    mat, n_features = 100, normalize_mass = TRUE
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
                strip_prefix(feature), loading, dim
            ),
            sign = if_else(loading >= 0, "positive", "negative")
        )

    ggplot(loadings, aes(feature_ordered, omic)) +
        geom_point(
            aes(size = abs(loading), fill = sign),
            shape = 22, color = "black", stroke = 0.4
        ) +
        facet_wrap(~dim, nrow = 2, scales = "free_x") +
        scale_x_reordered() +
        scale_size_area(max_size = 10, name = "|Loading|") +
        scale_fill_manual(
            values = c(positive = "black", negative = "white"),
            name = "Sign"
        ) +
        labs(
            title = "TCGA Loadings",
            x = "Feature", y = "Omic"
        ) +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, size = 5),
            panel.background = element_rect(fill = "#cecdcd", color = NA),
            panel.grid.major = element_blank(),
            strip.text = element_text(size = 10)
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

# Build one vignette panel: importance-mass bar chart + PDP line plot
vignette_panel <- function(feat, importance_mass, vignette_pdp) {
    feat_scores <- importance_mass |>
        filter(feature == feat) |>
        mutate(method = fct_reorder(method, mass))

    p_bar <- ggplot(feat_scores, aes(mass, method)) +
        geom_col() +
        scale_x_continuous(labels = scales::label_percent(accuracy = 0.1)) +
        labs(
            x = "Fraction of total |importance| mass", y = NULL,
            title = strip_prefix(feat)
        )

    pdp_sub <- vignette_pdp |> filter(feature == feat)
    p_pdp <- ggplot(pdp_sub, aes(grid_value, pdp_value)) +
        geom_line(linewidth = 1) +
        labs(x = strip_prefix(feat), y = "Partial dependence")

    p_bar | p_pdp
}

# Full vignette figure: top-k features as stacked panels
vignette_plot <- function(importance, method_cols, k = 3) {
    importance_mass <- normalize_importance_mass(importance, method_cols)
    vp <- read_csv(path(res, "vignette_pdp.csv"), show_col_types = FALSE) |>
        filter(feature %in% importance_mass$feature)
    top_feats <- head(unique(vp$feature), k)

    panels <- map(top_feats, \(f) vignette_panel(f, importance_mass, vp))
    wrap_plots(panels, ncol = 1)
}

# --- Run ---------------------------------------------------------------------

# Load & prepare data
d <- load_importance_data(sweep_sample_size())

# Figure 1 — PCA bi-plot
ggsave(
    path(res, "fig_method_pca.pdf"),
    pca_plot(d$mat, normalize_mass = pca_normalize_mass),
    width = 5, height = 3.5, dpi=400
)

# Figure 2 — PCA feature loadings
ggsave(
    path(res, "fig_method_pca_loadings.pdf"),
    pca_loadings_plot(d$mat, normalize_mass = pca_normalize_mass),
    width = 9, height = 4
)

# Figure 3 — Spearman correlation heatmap
ggsave(
    path(res, "fig_method_corr.pdf"),
    corr_plot(d$mat),
    width = 8, height = 7
)

# Figure 4 — Vignette panels
ggsave(
    path(res, "fig_vignettes.pdf"),
    vignette_plot(d$importance, d$method_cols),
    width = 10, height = 12
)

# Figure 5 — SAGE vs minSHAP credit-splitting scatter
ggsave(
    path(res, "fig_credit_splitting.pdf"),
    credit_splitting_plot(d$importance),
    width = 8, height = 6
)
