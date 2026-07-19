# Visualize relationship between synthetic data explanations
#
# These are helper functions used in case_studies/sweep_tabular/visualize.qmd.

library(tidyverse)
library(glue)
library(scico)
library(FactoMineR)
library(fs)

#' Fixed feature order to use across synthetic datasets
FEATURE_ORDER <- c(paste0("x", 1:4), paste0("noise_", 1:6))

#' Labels to describe what type of null each variable encodes
#'
#' This will depend on choices in the configuration. So not general purpose, but
#' should be enough for making the visualizations.
#' @return tibble(dataset, feature, null_type)
null_type_table <- function() {
    noise_cols <- paste0("noise_", 1:6)
    noise_rows <- \(dataset) {
        tibble(dataset = dataset, feature = noise_cols, null_type = "Null")
    }

    bind_rows(
        tibble(
            dataset = "linear_additive",
            feature = paste0("x", 1:4),
            null_type = "Signal"
        ),
        noise_rows("linear_additive"),
        tibble(
            dataset = "xor",
            feature = paste0("x", 1:4),
            null_type = "Marginal null"
        ),
        noise_rows("xor"),
        tibble(
            dataset = "product_interaction",
            feature = paste0("x", 1:4),
            null_type = "Marginal null"
        ),
        noise_rows("product_interaction"),
        tibble(
            dataset = "dependent_features",
            feature = c("x1", "x2", "x3", "x4"),
            null_type = c(
                "Signal", "Conditional null", "Signal", "Conditional null"
            )
        ),
        noise_rows("dependent_features"),
        tibble(
            dataset = "confounding",
            feature = paste0("x", 1:4),
            null_type = "Causal null"
        ),
        noise_rows("confounding"),
        tibble(
            dataset = "quadratic",
            feature = paste0("x", 1:4),
            null_type = "Marginal null"
        ),
        noise_rows("quadratic")
    )
}

#' Null-type color scale
#' @param null_types character vector of levels in the order we want them to
#'   appear in the legend
null_type_colors <- function(
    null_types = c(
        "Signal", "Marginal null", "Conditional null", "Causal null", "Null"
    )
) {
    setNames(scico(length(null_types), palette = "berlin"), null_types)
}

#' Load method's variable importance CSV.
#' @param dataset dataset name
#' @param n sample size
#' @param response_type "classification" or "regression"
#' @param seed seed
#' @param method method name
#' @param results_dir results directory
load_method_importance <- function(
    dataset, n, response_type, seed, method, results_dir = "results"
) {
    path <- path(
        results_dir, glue("{dataset}_{n}_{response_type}_{seed}_{method}.csv")
    )
    read_csv(path, show_col_types = FALSE) |>
        mutate(method = method)
}

#' Importance matrix for one (dataset, n, response_type, seed).
#' @param dataset dataset name
#' @param n sample size
#' @param response_type "classification" or "regression"
#' @param seed seed
#' @param methods method names
#' @param results_dir results directory
importance_matrix <- function(
    dataset, n, response_type, seed, methods, results_dir = "results"
) {
    map_dfr(
        methods,
        ~ load_method_importance(
            dataset, n, response_type, seed, .x, results_dir
        )
    ) |>
        pivot_wider(names_from = method, values_from = importance) |>
        column_to_rownames("feature")
}

#' Multiple Factor Analysis of importance profiles, blocked by sample size.
#'
#' The rows in each table are features, the columns are explanation methods. By
#' blocking across seeds within each sample size, we capture cross-method
#' relationships while accounting for seed variation.
#'
#' @param dataset dataset name
#' @param ns sample sizes (default c(50, 500, 5000))
#' @param response_type response type (default "classification")
#' @param seeds seeds
#' @param methods method names
#' @param results_dir results directory
#' @return list(coords, partial, var_explained, loadings)
mfa_by_sample_size <- function(
    dataset, ns = c(50, 500, 5000), response_type = "classification",
    seeds, methods, results_dir = "results"
) {
    configs <- expand_grid(n = ns, seed = seeds)
    mats <- pmap(
        configs,
        \(n, seed) importance_matrix(
            dataset, n, response_type, seed, methods, results_dir
        )
    )
    block_names <- pmap_chr(configs, function(n, seed) glue("n{n}_s{seed}"))
    feature_order <- rownames(mats[[1]])

    combined <- reduce(mats, cbind)
    colnames(combined) <- glue(
        "{rep(block_names, each = length(methods))}",
        "_{rep(methods, times = length(block_names))}"
    )

    mfa <- MFA(
        as.data.frame(combined),
        group = rep(length(methods), length(block_names)),
        type = rep("s", length(block_names)),
        name.group = block_names,
        graph = FALSE
    )

    coords <- as_tibble(mfa$ind$coord[, 1:2], rownames = "feature")
    var_explained <- mfa$eig[1:2, 2]

    # FactoMineR joins the individual name and group name with "." (e.g.
    # "x1.n50_s2026"), not "_" -- the group name itself keeps its own "_".
    partial <- as_tibble(mfa$ind$coord.partiel[, 1:2], rownames = "row") |>
        extract(
            row,
            into = c("feature", "n", "seed"),
            regex = "^(.+)[.]n([0-9]+)_s([0-9]+)$"
        ) |>
        mutate(across(c(n, seed), as.numeric)) |>
        group_by(feature, n) |>
        summarize(Dim.1 = mean(Dim.1), Dim.2 = mean(Dim.2), .groups = "drop")

    loadings <- as_tibble(mfa$quanti.var$coord[, 1:2], rownames = "variable") |>
        extract(
            variable,
            into = c("block", "method"),
            regex = "^(n[0-9]+_s[0-9]+)_(.+)$"
        ) |>
        group_by(method) |>
        summarize(Dim.1 = mean(Dim.1), Dim.2 = mean(Dim.2), .groups = "drop")

    list(
        coords = coords, partial = partial, var_explained = var_explained,
        loadings = loadings
    )
}

#' Cross-method importance ranks per feature.
#' Ranks methods' importance within (dataset, n, response_type, seed).
#' @param dataset dataset name
#' @param n sample size
#' @param response_type "classification" or "regression"
#' @param seed seed
#' @param methods method names
#' @param results_dir results directory
method_ranks <- function(
    dataset, n, response_type, seed, methods, results_dir = "results"
) {
    map_dfr(
        methods,
        ~ load_method_importance(
            dataset, n, response_type, seed, .x, results_dir
        )
    ) |>
        group_by(method) |>
        mutate(rank = rank(-importance, ties.method = "average")) |>
        ungroup()
}


bump_panel <- function(ds) {
    df <- ranks_by_dataset[[ds]]
    labels <- null_type_table() |>
        filter(dataset == ds) |>
        mutate(x = match(feature, FEATURE_ORDER)) |>
        arrange(x)
    df <- df |>
        left_join(select(labels, feature, x, null_type), by = "feature")

    ggplot(df, aes(x = x, y = rank)) +
        geom_tile(
            data = labels, aes(x = x, y = strip_y, fill = null_type),
            height = 0.7, width = 0.9, inherit.aes = FALSE
        ) +
        geom_bump(
            aes(group = interaction(method, seed)),
            color = axiom_palette$grid, linewidth = 0.4, alpha = 0.35,
            smooth = 8
        ) +
        geom_point(
            aes(shape = method, color = null_type),
            size = 1.8, stroke = 0.8, alpha = 0.35
        ) +
        scale_x_continuous(breaks = labels$x, labels = labels$feature) +
        scale_y_reverse(breaks = 1:n_features, limits = c(strip_y + 0.6, 0.5)) +
        scale_shape_manual(
            values = method_shapes, limits = methods,
            name = "Method", guide = "none"
        ) +
        scale_color_manual(
            values = null_colors, limits = names(null_colors), guide = "none"
        ) +
        facet_grid(. ~ method) +
        scale_fill_manual(
            values = null_colors, limits = names(null_colors), guide = "none"
        ) +
        theme(axis.text.x = element_text(angle = 90, size = 7)) +
        labs(title = ds, x = NULL, y = "Rank")
}