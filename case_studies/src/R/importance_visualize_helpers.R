# Visualize relationship between synthetic data explanations
#
# These are helper functions used in case_studies/sweep_tabular/visualize.qmd.

library(tidyverse)
library(glue)
library(here)
library(scico)
library(FactoMineR)
library(fs)
library(yaml)

#' Vector giving the feature orders (signal then noise features)
#'
#' This is all read from the configuration file.
#'
#' @param config_path sweep_tabular's config.yaml
sweep_feature_order <- function(
    config_path = path(here("case_studies", "sweep_tabular"), "config.yaml")
) {
    dimensions <- yaml::read_yaml(config_path)$dimensions
    c(
        paste0("x", seq_len(dimensions$n_nonnull)),
        paste0("noise_", seq_len(dimensions$n_features - dimensions$n_nonnull))
    )
}

FEATURE_ORDER <- sweep_feature_order()

#' Labels to describe what type of null each variable encodes
#'
#' @param evaluation_dir directory holding null_labels.csv
#' @return tibble(dataset, feature, null_type)
null_type_table <- function(
    evaluation_dir = path(here("case_studies", "sweep_tabular"), "results", "evaluation")
) {
    labels_path <- path(evaluation_dir, "null_labels.csv")
    if (!file_exists(labels_path)) {
        stop(glue(
            "{labels_path} not found. Run `python case_studies/sweep_tabular/evaluate.py` ",
            "to derive the null-type ground truth from datasets.py."
        ))
    }

    read_csv(labels_path, show_col_types = FALSE) |>
        select(dataset, feature, null_type = null_label)
}

#' Null-type color scale
#' @param null_types character vector of levels in the order we want them to
#'   appear in the legend
null_type_colors <- function(
    null_types = c(
        "Signal", "Marginal null", "Conditional null", "Risk null",
        "Functional null", "Causal null", "Null"
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

# minSHAP mediated-chain diagnostic --------------------------------------

#' Load and wrangle one run's sampled mediated-chain contribution details.
#'
#' @param stem file stem, e.g. "mediated_chains_5000_regression_2026"
#' @param results_dir results directory
#' @param feature_order feature display order (default FEATURE_ORDER)
#' @return list(contributions, membership)
load_mediated_contributions <- function(
    stem, results_dir, feature_order = FEATURE_ORDER
) {
    contributions <- read_csv(
        path(results_dir, glue("{stem}_risk_contributions.csv")),
        show_col_types = FALSE
    ) |>
        mutate(
            target = factor(target, levels = feature_order),
            ordering_sorted = reorder_within(ordering, contribution, target)
        )

    membership_cols <- paste0("in_", feature_order)
    membership <- contributions |>
        select(ordering_sorted, target, all_of(membership_cols)) |>
        pivot_longer(
            all_of(membership_cols),
            names_to = "coalition_feature",
            names_prefix = "in_",
            values_to = "included"
        ) |>
        mutate(
            coalition_feature = factor(coalition_feature, levels = feature_order)
        )

    list(contributions = contributions, membership = membership)
}

#' Subset and re-level the mediated-chain tables to a chosen set of targets.
#'
#' Shared by the full and condensed contribution-focus figures: each fixes a
#' set of targets (features being added) and needs matching membership, bar,
#' and coalition-outline tables to plot them.
#'
#' @param membership `load_mediated_contributions()$membership`
#' @param contributions `load_mediated_contributions()$contributions`
#' @param targets target features to keep, in the order they should be
#'   faceted
#' @param feature_order feature display order for the coalition_feature axis
#'   (default FEATURE_ORDER)
#' @return list(membership, bars, outline)
mediated_focus_data <- function(
    membership, contributions, targets, feature_order = FEATURE_ORDER
) {
    membership <- membership |>
        filter(target %in% targets) |>
        mutate(target = factor(target, levels = targets))
    bars <- contributions |>
        filter(target %in% targets) |>
        mutate(
            target = factor(target, levels = targets),
            contribution_sign = if_else(contribution >= 0, "positive", "negative")
        )
    outline <- membership |>
        distinct(ordering_sorted, target) |>
        mutate(
            coalition_feature = factor(as.character(target), levels = feature_order)
        )
    list(membership = membership, bars = bars, outline = outline)
}

#' Coalition-membership tile plot for the mediated-chain diagnostic.
#'
#' Grey cells mark the predecessor coalition S; the orange outline marks the
#' target feature j itself.
#'
#' @param membership one panel per target's coalition membership, see
#'   `mediated_focus_data()$membership`
#' @param outline rows to outline, see `mediated_focus_data()$outline`
#' @param subtitle plot subtitle
#' @param title plot title
mediated_membership_plot <- function(
    membership, outline,
    title = "Coalitions for mediated chains"
) {
    membership |>
        filter(!str_detect(coalition_feature, "^noise_([2-9]|[1-9][0-9]+)$")) |>
        ggplot(aes(coalition_feature, ordering_sorted)) +
        geom_tile(
            aes(fill = factor(included)), color = "white", linewidth = 0.25
        ) +
        geom_tile(
            data = outline, fill = NA, color = "#bf3600", linewidth = 0.9
        ) +
        facet_wrap(~target, ncol = 3, scales = "free_y") +
        scale_fill_manual(
            values = c(`0` = "#f5f5f5", `1` = "#737373"),
            labels = c(`0` = "Absent", `1` = "In S"),
            name = "Coalition"
        ) +
        scale_y_reordered() +
        labs(title = title, x = "Feature", y = "Sampled ordering") +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1, size = 10),
            strip.text = element_text(size = 12),
            legend.text = element_text(size = 12),
            legend.title = element_text(size = 14),
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            title.text = element_text(size = 16),
            axis.title = element_text(size = 14),
            panel.grid = element_blank()
        )
}

#' Edge-term bar plot for the mediated-chain diagnostic.
#'
#' Bars show the actual sampled I_j(S) = V(S union j) - V(S), one panel per
#' target j.
#'
#' @param bars one panel per target's contributions, see
#'   `mediated_focus_data()$bars`
mediated_bars_plot <- function(bars) {
    ggplot(bars, aes(contribution, ordering_sorted, fill = contribution_sign)) +
        geom_col(width = 0.75, color = axiom_palette$ink) +
        geom_vline(xintercept = 0, color = axiom_palette$ink, linewidth = 0.35) +
        facet_wrap(~target, ncol = 3, scales = "free_y") +
        scale_fill_manual(
            values = c(positive = "#020202", negative = "#f3f3f3"),
            guide = "none"
        ) +
        scale_y_reordered() +
        labs(
            title = expression("Contributions " * I[j](S)),
            x = expression(I[j](S)), y = NULL,
        ) +
        theme(
            axis.text.y = element_blank(),
            strip.text = element_text(size = 12),
            title.text = element_text(size = 16),
            axis.ticks.y = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_rect(fill = "#ffffff")
        )
}
