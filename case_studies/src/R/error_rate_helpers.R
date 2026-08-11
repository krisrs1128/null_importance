# Null importance as a testing problem.
#
# Helpers for case_studies/sweep_tabular/error_rates.qmd. They read the tables
# written by sweep_tabular/evaluate.py; they do not recompute truth or rates.
library(tidyverse)
library(glue)
library(fs)

NOTION_ORDER <- c("marginal", "conditional", "risk", "functional", "causal")
SYMBOL_LABELS <- c(check = "✓", tilde = "~", cross = "×")

#: Low, middle, and high anchors for every scale in error_rates.qmd.
RATE_COLORS <- c("#5c538e", "#efc4b9", "#5eb978")

#' Discrete colour scale interpolating the three anchors.
#'
#' Used where the mapped variable is a method or a null notion, so the number
#' of levels is only known at draw time.
scale_color_rate_d <- function(...) {
    discrete_scale(
        "colour", palette = \(n) colorRampPalette(RATE_COLORS)(n), ...
    )
}

#' Load the evaluation tables written by evaluate.py.
#'
#' @param evaluation_dir Directory containing the evaluation CSVs.
#' @return A named list of tibbles. `notion` and `dataset_id` are ordered
#'   factors, so facets follow the paper's D1--D7 order.
load_evaluation <- function(evaluation_dir) {
    read_one <- function(name) {
        frame <- read_csv(path(evaluation_dir, name), show_col_types = FALSE)
        if ("notion" %in% names(frame)) {
            frame <- mutate(
                frame,
                notion = factor(notion, levels = NOTION_ORDER)
            )
        }
        if ("dataset_id" %in% names(frame)) {
            frame <- mutate(
                frame,
                dataset_id = factor(dataset_id, levels = paste0("D", 1:7))
            )
        }
        frame
    }

    list(
        truth = read_one("null_truth.csv"),
        labels = read_one("null_labels.csv"),
        scores = read_one("scores_long.csv"),
        confusion = read_one("confusion.csv"),
        rates = read_one("error_rates.csv"),
        pooled = read_one("error_rates_pooled.csv"),
        sweep = read_one("threshold_sweep.csv"),
        curves = read_one("curve_summary.csv"),
        table1 = read_one("table1_empirical.csv")
    )
}

#' Put the four cells of each 2x2 table in long form.
#'
#' @param confusion Confusion tibble from `load_evaluation`.
#' @param notion Null notion to display.
#' @param null_scope `"signal"` for structural features only, or `"all"`.
#' @return A tibble with `dataset_id`, `method`, `truth`, `decision`, and
#'   `count`.
confusion_cells <- function(confusion, notion, null_scope = "signal") {
    confusion |>
        filter(notion == !!notion, null_scope == !!null_scope) |>
        group_by(dataset_id, method) |>
        summarise(across(c(tn, fp, fn, tp), sum), .groups = "drop") |>
        pivot_longer(
            c(tn, fp, fn, tp),
            names_to = "cell",
            values_to = "count"
        ) |>
        mutate(
            truth = factor(
                if_else(cell %in% c("tn", "fp"), "null", "non-null"),
                levels = c("null", "non-null")
            ),
            decision = factor(
                if_else(
                    cell %in% c("fp", "tp"),
                    "called non-null", "called null"
                ),
                levels = c("called null", "called non-null")
            ),
            is_error = cell %in% c("fp", "fn")
        )
}

#' Tile 2x2 tables for one notion across datasets and methods.
#'
#' The off-diagonal cells are outlined because they are the two error types.
#'
#' @param confusion Confusion tibble.
#' @param notion Null notion to display.
#' @param null_scope `"signal"` or `"all"`.
confusion_panel <- function(confusion, notion, null_scope = "signal") {
    cells <- confusion_cells(confusion, notion, null_scope)

    ggplot(cells, aes(decision, fct_rev(truth))) +
        geom_tile(aes(fill = count), color = NA) +
        geom_tile(
            data = filter(cells, is_error),
            fill = NA, color = axiom_palette$ink, linewidth = 0.4
        ) +
        geom_text(aes(label = count), size = 2.6, color = axiom_palette$ink) +
        facet_grid(method ~ dataset_id, switch = "y") +
        scale_fill_gradientn(colours = RATE_COLORS, name = "features") +
        labs(
            x = NULL, y = NULL,
            title = glue("{str_to_title(notion)} null: calls and ground truth"),
            subtitle = paste(
                "Outlined cells are errors;", "lower right is type I,",
                "upper left is type II"
            )
        ) +
        theme(
            axis.text.x = element_text(angle = 30, hjust = 1),
            strip.text.y.left = element_text(angle = 0),
            panel.grid.major = element_blank()
        )
}

#' Draw a method-by-dataset heatmap for one rate.
#'
#' @param rates Pooled error-rate tibble.
#' @param metric `"fpr"` or `"power"`.
#' @param null_scope `"signal"` or `"all"`.
#' @param midpoint Value at the centre of the diverging scale, such as the
#'   target alpha.
error_rate_heatmap <- function(rates, metric = "fpr", null_scope = "signal",
                               midpoint = 0.1) {
    rates |>
        filter(null_scope == !!null_scope) |>
        ggplot(aes(factor(n), method)) +
        geom_tile(aes(fill = .data[[metric]])) +
        facet_grid(notion ~ dataset_id + response_type, switch = "y") +
        scale_fill_gradientn(
            colours = RATE_COLORS,
            rescaler = ~ scales::rescale_mid(.x, mid = midpoint),
            na.value = axiom_palette$grid, name = metric
        ) +
        labs(
            x = "sample size", y = NULL,
            title = paste(
                if (metric == "fpr") "Type I error" else "Power",
                "by method, dataset, and sample size"
            ),
            subtitle = paste(
                "Grey cells are undefined; the colour scale is centred at",
                midpoint
            )
        ) +
        theme(
            strip.text.y.left = element_text(angle = 0),
            strip.text.x = element_text(size = rel(0.7)),
            panel.grid.major = element_blank()
        )
}

#' Draw precision-recall curves over the decision threshold.
#'
#' @param sweep Threshold-sweep tibble.
#' @param notion Null notion to display.
#' @param null_scope `"signal"` or `"all"`.
pr_curve_panel <- function(sweep, notion, null_scope = "signal") {
    sweep |>
        filter(notion == !!notion, null_scope == !!null_scope) |>
        group_by(dataset_id, method, threshold) |>
        summarise(
            across(c(precision, power), ~ mean(.x, na.rm = TRUE)),
            .groups = "drop"
        ) |>
        filter(!is.na(precision), !is.na(power)) |>
        ggplot(aes(power, precision, color = method)) +
        geom_path(linewidth = 0.5) +
        geom_point(size = 0.7) +
        facet_wrap(~dataset_id, nrow = 2) +
        scale_color_rate_d() +
        lims(x = c(0, 1), y = c(0, 1)) +
        labs(
            x = "recall (power)", y = "precision",
            title = glue("Precision-recall: {notion} null"),
            subtitle = "Each line shows one method as the threshold changes"
        )
}

#' Draw threshold-free ranking performance.
#'
#' @param curves Curve-summary tibble.
#' @param null_scope `"signal"` or `"all"`.
roc_panel <- function(curves, null_scope = "signal") {
    curves |>
        filter(null_scope == !!null_scope, !is.na(roc_auc)) |>
        group_by(dataset_id, method, notion) |>
        summarise(roc_auc = mean(roc_auc), .groups = "drop") |>
        ggplot(aes(roc_auc, method)) +
        geom_vline(
            xintercept = 0.5, color = axiom_palette$grid, linewidth = 0.4
        ) +
        geom_point(aes(color = notion), size = 1.6, alpha = 0.85) +
        facet_wrap(~dataset_id, nrow = 2) +
        scale_color_rate_d() +
        labs(
            x = "ROC AUC (0.5 = no separation)", y = NULL,
            title = "Ranking quality without a threshold",
            subtitle = paste(
                "Points coincide where the DGP gives two notions",
                "the same labels"
            )
        )
}

#' Plot the empirical counterpart to Table 1.
#'
#' @param table1 Empirical table tibble.
table1_comparison_plot <- function(table1) {
    plot_data <- table1 |>
        mutate(label = SYMBOL_LABELS[symbol])

    ggplot(plot_data, aes(notion, method)) +
        geom_tile(
            aes(fill = controlled_fraction),
            color = axiom_palette$bg, linewidth = 1
        ) +
        geom_text(aes(label = label), size = 4, color = axiom_palette$ink) +
        scale_fill_gradientn(
            colours = RATE_COLORS, limits = c(0, 1),
            name = "fraction of cells\nwith type I control"
        ) +
        labs(
            x = NULL, y = NULL,
            title = "Empirical counterpart to Table 1",
            subtitle = paste(
                "✓ controlled and powered, ~ one of the two,",
                "× control fails in most cells"
            )
        ) +
        theme(panel.grid.major = element_blank())
}

#' Load Table 1 and relabel its rows to the swept method names.
#'
#' @param path CSV containing the transcription.
#' @param theoretical_rows Named list mapping a method name to a CSV row name.
load_theoretical_table <- function(path, theoretical_rows) {
    mapping <- theoretical_rows |>
        compact() |>
        imap_dfr(~ tibble(method = .y, row = as.character(.x)))

    read_csv(path, show_col_types = FALSE) |>
        pivot_longer(-method, names_to = "notion", values_to = "symbol") |>
        rename(row = method) |>
        inner_join(mapping, by = "row") |>
        mutate(notion = factor(notion, levels = NOTION_ORDER)) |>
        select(method, notion, symbol)
}
