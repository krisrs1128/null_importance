# Null importance as a testing problem.
#
# Helpers for case_studies/sweep_tabular/error_rates.qmd. They read the tables
# written by sweep_tabular/evaluate.py; they do not recompute truth or rates.
library(tidyverse)
library(glue)
library(fs)
library(patchwork)
library(scico)

NOTION_ORDER <- c("marginal", "conditional", "risk", "functional", "causal")
SYMBOL_LABELS <- c(check = "✓", tilde = "~", cross = "×")
SYMBOL_RANK <- c(cross = 0, tilde = 1, check = 2)
STATUS_COLORS <- c("#ffffff", "#e3e3e3", "#797979")

#: See definition in config.yaml
DATASET_IDS <- glue("D{1:10}")


#' Discrete colour scale interpolating the status levels anchors.
scale_color_rate_d <- function(...) {
    discrete_scale(
        "colour", palette = \(n) colorRampPalette(STATUS_COLORS)(n), ...
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
                dataset_id = factor(dataset_id, levels = DATASET_IDS)
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
        table1 = read_one("table1_empirical.csv"),
        table1_by_n = read_one("table1_empirical_by_n.csv"),
        null_mass = read_one("null_mass.csv")
    )
}

#' Restrict every table that carries a `response_type` column to one type.
#'
#' This is useful for defining summaries among just the classification (or just
#' the regression) models.
#' @param ev Named list of tibbles from `load_evaluation`.
#' @param response_type Value to keep, e.g. `"regression"`.
filter_response_type <- function(ev, response_type) {
    map(ev, \(frame) {
        if ("response_type" %in% names(frame)) {
            filter(frame, response_type == !!response_type)
        } else {
            frame
        }
    })
}

#' Count the features on which each pair of notions disagrees.
#'
#' For how many data sets do the notions of null importance coincide versus
#' disagree?
#'
#' @param truth Ground-truth tibble from `load_evaluation`.
#' @return A tibble with `row`, `column`, and `n_differ`.
notion_disagreements <- function(truth) {
    structural <- filter(truth, null_kind != "pad")
    null_set <- function(which_notion) {
        structural |>
            filter(notion == which_notion, is_null) |>
            distinct(dataset, feature)
    }

    expand_grid(row = NOTION_ORDER, column = NOTION_ORDER) |>
        mutate(
            n_differ = map2_int(row, column, \(a, b) {
                left <- null_set(a)
                right <- null_set(b)
                nrow(setdiff(left, right)) + nrow(setdiff(right, left))
            }),
            row = factor(row, levels = NOTION_ORDER),
            column = factor(column, levels = NOTION_ORDER)
        )
}

#' Tile the pairwise notion disagreements.
#'
#' @param truth Ground-truth tibble from `load_evaluation`.
notion_separation_panel <- function(truth) {
    counts <- notion_disagreements(truth) |>
        filter(row != column)

    ggplot(counts, aes(column, fct_rev(row))) +
        geom_tile(aes(fill = n_differ), color = axiom_palette$bg) +
        geom_text(aes(
            label = n_differ), size = 3.4, color = axiom_palette$ink
        ) +
        scale_x_discrete(expand = c(0, 0)) +
        scale_y_discrete(expand = c(0, 0)) +
        scale_fill_viridis_c(
            limits = c(0, NA), name = "features whose\nlabels differ"
        ) +
        labs(
            x = NULL, y = NULL,
            title = "How do notions differ?"
        ) +
        theme(panel.grid.major = element_blank())
}

#' Pivot the 2x2 confusion tables into long form.
#'
#' Note that this will sum across all seeds at each sample size.
#'
#' @param confusion Confusion tibble from `load_evaluation`.
#' @param notion Null notion to display.
#' @return A tibble with `dataset_id`, `method`, `n`, `truth`, `decision`, and
#'   `count`.
confusion_cells <- function(confusion, notion) {
    confusion |>
        filter(notion == !!notion) |>
        group_by(dataset_id, method, n) |>
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
#' Methods are ordered according to overall F1.
#'
#' @param confusion Confusion tibble.
#' @param notion Null notion to display.
#' @param sample_size If given, restrict to this `n` instead of pooling every
#'   sample size into one count. Small- and large-`n` behaviour can differ
#'   enough that pooling hides it, so this is not a default.
confusion_panel <- function(confusion, notion, sample_size = NULL) {
    # Prepare the confusion counts to display.
    plot_data <- confusion_cells(confusion, notion) |>
        filter(is.null(sample_size) | n %in% sample_size) |>
        summarise(
            count = sum(count),
            .by = c(dataset_id, method, cell, truth, decision, is_error)
        )

    # Order methods by overall F1.
    method_scores <- plot_data |>
        summarise(count = sum(count), .by = c(method, cell)) |>
        pivot_wider(
            names_from = cell,
            values_from = count,
            values_fill = 0
        ) |>
        mutate(f1 = 2 * tp / (2 * tp + fp + fn))

    method_order <- method_scores |>
        arrange(desc(f1)) |>
        pull(method)

    plot_data <- plot_data |>
        mutate(method = factor(method, levels = method_order))

    sample_label <- ifelse(
        is.null(sample_size), "", glue(" [n = {sample_size}]")
    )

    # Draw the panel.
    ggplot(plot_data, aes(decision, fct_rev(truth))) +
        geom_tile(aes(fill = count), color = NA) +
        geom_tile(
            data = filter(plot_data, is_error),
            fill = NA, color = axiom_palette$ink, linewidth = 0.4
        ) +
        geom_text(aes(label = count), size = 2.6, color = axiom_palette$ink) +
        facet_grid(method ~ dataset_id, switch = "y") +
        scale_fill_continuous(palette = c("white", "#8f8f8f")) +
        labs(
            x = NULL, y = NULL,
            title = glue("{str_to_title(notion)} null{sample_label}")
        ) +
        theme(
            axis.text.x = element_text(angle = 90, hjust = 1),
            strip.text.y.left = element_text(angle = 0),
            panel.grid.major = element_blank()
        )
}

#' Plot the empirical counterpart to Table 1.
#'
#' Methods are ordered bottom-to-top by mean FPR across notions, best (lowest)
#' first, so the best-controlled method sits at the top of the tile.
#'
#' @param table1 Empirical table tibble.
table1_comparison_plot <- function(table1) {
    fpr_order <- table1 |>
        summarise(mean_fpr = mean(mean_fpr, na.rm = TRUE), .by = method) |>
        arrange(desc(mean_fpr)) |>
        pull(method)

    plot_data <- table1 |>
        mutate(
            label = SYMBOL_LABELS[symbol],
            method = factor(method, levels = fpr_order)
        )

    ggplot(plot_data, aes(notion, method)) +
        geom_tile(
            aes(fill = controlled_fraction),
            color = axiom_palette$bg, linewidth = 1
        ) +
        geom_text(aes(label = label), size = 4, color = axiom_palette$ink) +
        scale_x_discrete(expand = c(0, 0)) +
        scale_y_discrete(expand = c(0, 0)) +
        scale_fill_scico(
            palette = "vik", limits = c(0, 1),
            direction = -1,
            name = "fraction of cells\nwith type I control"
        ) +
        labs(
            x = NULL, y = NULL,
            title = "Empirical version of Table 1"
        ) +
        theme(panel.grid.major = element_blank())
}

#' Plot Table 1 showing power in each cell.
#'
#' Methods are ordered bottom-to-top by mean power across notions, best
#' (highest) first, so the most powerful method sits at the top of the tile.
#'
#' @param table1 Empirical table tibble.
table1_power_plot <- function(table1) {
    power_order <- table1 |>
        summarise(mean_power = mean(mean_power, na.rm = TRUE), .by = method) |>
        arrange(mean_power) |>
        pull(method)

    plot_data <- table1 |>
        mutate(
            label = SYMBOL_LABELS[symbol],
            method = factor(method, levels = power_order)
        )

    ggplot(plot_data, aes(notion, method)) +
        geom_tile(
            aes(fill = powered_fraction),
            color = axiom_palette$bg, linewidth = 1
        ) +
        geom_text(aes(label = label), size = 4, color = axiom_palette$ink) +
        scale_x_discrete(expand = c(0, 0)) +
        scale_y_discrete(expand = c(0, 0)) +
        scale_fill_scico(
            palette = "bam", limits = c(0, 1),
            name = "fraction of cells\nwith power"
        ) +
        labs(
            x = NULL, y = NULL,
            title = "Power in each cell"
        ) +
        theme(panel.grid.major = element_blank())
}

#' Side-by-side comparison of type I control and power.
#'
#' @param table1 Empirical table tibble.
#' @param sample_size If given, restrict to this `n` instead of the version
#'   pooled over sample size. Requires `table1` to carry an `n` column, i.e.
#'   `ev$table1_by_n` rather than the pooled `ev$table1`.
table1_comparison_combined <- function(table1, sample_size = NULL) {
    if (!is.null(sample_size)) {
        table1 <- filter(table1, n == sample_size)
    }

    p_control <- table1_comparison_plot(table1) +
        labs(title = "Type I control in each cell")
    p_power <- table1_power_plot(table1)

    title_n <- if (is.null(sample_size)) "" else glue(", n = {sample_size}")
    p_control + p_power +
        plot_layout(guides = "collect") +
        plot_annotation(
            title = glue("Empirical counterpart to Table 1{title_n}"),
            theme = theme(
                plot.title = element_text(size = 14, face = "bold")
            )
        )
}

#' Rank methods by the mass they put on null features.
#'
#'  This part's the null_mass statistic defined in evaluate.py. Zero means the
#' method respects the notion's null set; one means all of its mass falls on
#' null features.
#'
#' @param null_mass Null-mass tibble from `load_evaluation`.
#' @param sample_size If given, restrict to this `n` rather than averaging over
#'   every sample size.
#' @return A ggplot, one point per dataset, faceted by notion.
null_mass_panel <- function(null_mass, sample_size = NULL) {
    if (!is.null(sample_size)) {
        null_mass <- filter(null_mass, n == sample_size)
    }

    # extract null mass values
    cells <- null_mass |>
        summarise(
            across(
                c(null_magnitude, nonnull_magnitude),
                \(x) mean(x, na.rm = TRUE)
            ),
            .by = c(method, notion, dataset_id, n)
        ) |>
        mutate(
            null_mass = null_magnitude / (null_magnitude + nonnull_magnitude),
            null_mass = if_else(is.finite(null_mass), null_mass, NA_real_)
        )

    # sort methods from best to worst
    method_order <- cells |>
        summarise(overall = mean(null_mass, na.rm = TRUE), .by = method) |>
        arrange(desc(overall)) |>
        pull(method)
    cells <- mutate(cells, method = factor(method, levels = method_order))

    # visualize
    title_n <- if (is.null(sample_size)) "" else glue(" [n = {sample_size}]")
    ggplot(cells, aes(method, reorder(dataset_id, null_mass, na.rm = TRUE))) +
        geom_vline(
            xintercept = 0.5, linetype = "dashed", color = axiom_palette$grid
        ) +
        geom_tile(aes(fill = null_mass), size = 2, alpha = 0.85) +
        facet_wrap(~notion) +
        scale_fill_scico(
            palette = "berlin", midpoint = 0.2, na.value = axiom_palette$grid
        ) +
        labs(
            fill = "fraction of |phi| mass on null features",
            y = NULL, x = NULL,
            title = glue("Mass placed on null features{title_n}")
        ) +
        theme(
            panel.grid.major.y = element_blank(),
            axis.text.x = element_text(angle = 90, hjust = 1)
        )
}

#' Compare the transcribed Table 1 with what the sweep observed.
#'
#' The glyph reads "theory then observation". Disagreements are the point of the
#' figure: a cell where the sweep is weaker than the table means the guarantee
#' did not survive contact with these data-generating mechanisms, and a cell
#' where it is stronger usually means no dataset exercises that notion hard
#' enough to expose the failure.
#'
#' @param table1 Empirical table tibble.
#' @param theoretical Tibble from `load_theoretical_table`.
theory_vs_empirical_panel <- function(table1, theoretical) {
    # create dataset to visualize
    joined <- theoretical |>
        inner_join(
            select(table1, method, notion, symbol),
            by = c("method", "notion"),
            suffix = c("_theory", "_empirical")
        ) |>
        mutate(
            label = str_c(
                SYMBOL_LABELS[symbol_theory], "→",
                SYMBOL_LABELS[symbol_empirical]
            ),
            agreement = factor(
                case_when(
                    symbol_theory == symbol_empirical ~ "agrees",
                    SYMBOL_RANK[symbol_empirical] >
                        SYMBOL_RANK[symbol_theory] ~ "observed is stronger",
                    TRUE ~ "observed is weaker"
                ),
                levels = c("observed is weaker", "agrees", "observed is stronger")
            )
        )

    ggplot(joined, aes(notion, method)) +
        geom_tile(aes(fill = agreement), color = axiom_palette$bg, linewidth = 1) +
        geom_text(aes(label = label), size = 3.4, color = axiom_palette$ink) +
        scale_fill_manual(
            values = setNames(
                STATUS_COLORS,
                c("observed is weaker", "agrees", "observed is stronger")
            ),
            name = NULL, drop = FALSE
        ) +
        labs(
            x = NULL, y = NULL,
            title = "Table 1 against the sweep"
        ) +
        theme(panel.grid.major = element_blank())
}

#' Check each claim in the `counterexamples` config block against the scores.
#'
#' One row per (claim, method). `rejection_rate` is the fraction of blocks in
#' which the method called the named features non-null, so `expected = "zero"`
#' asks for a low rate and `expected = "nonzero"` for a high one.
#'
#' @param scores Long score tibble from `load_evaluation` (`scores_long.csv`).
#' @param spec The `counterexamples` list read from config.yaml.
#' @param tolerance Slack allowed before a claim counts as failing.
counterexample_ledger <- function(scores, spec, tolerance = 0.25) {
    map_dfr(spec, \(entry) {
        features <- unlist(entry$features)
        observed <- scores |>
            filter(dataset == entry$dataset, feature %in% features) |>
            group_by(method) |>
            summarise(rejection_rate = mean(called_nonnull), .groups = "drop")

        bind_rows(
            tibble(method = unlist(entry$expect_zero), expected = "zero"),
            tibble(method = unlist(entry$expect_nonzero), expected = "nonzero")
        ) |>
            left_join(observed, by = "method") |>
            mutate(
                claim = entry$id,
                title = entry$title,
                dataset = entry$dataset,
                features = paste(features, collapse = ", "),
                null_under = paste(unlist(entry$null_under), collapse = ", "),
                holds = if_else(
                    expected == "zero",
                    rejection_rate <= tolerance,
                    rejection_rate >= 1 - tolerance
                )
            )
    }) |>
        select(
            claim, title, dataset, features, null_under, method, expected,
            rejection_rate, holds
        )
}

#' Per-feature scores for one dataset and method, split by decision.
#'
#' Signal features come first (`x1`, `x2`, ...), then pads (`noise_1`,
#' `noise_2`, ...).
#'
#' @param scores Long score tibble from `load_evaluation` (`scores_long.csv`).
#' @param which_dataset_id Paper identifier, e.g. `"D1"`.
#' @param which_method Method name to filter to.
feature_score_panel <- function(scores,
                                which_dataset_id,
                                which_method = "minshap") {
    plot_data <- filter(
        scores, dataset_id == which_dataset_id, method == which_method
    )

    n_signal <- n_distinct(plot_data$feature[plot_data$null_kind == "signal"])
    n_pad <- n_distinct(plot_data$feature[plot_data$null_kind == "pad"])
    feature_levels <- c(
        paste0("x", seq_len(n_signal)), paste0("noise_", seq_len(n_pad))
    )
    plot_data <- mutate(
        plot_data,
        feature = factor(feature, levels = feature_levels)
    )

    ggplot(plot_data, aes(feature, importance)) +
        geom_boxplot(
            outlier.shape = NA, fill = NA, color = axiom_palette$grid
        ) +
        geom_jitter(
            aes(color = called_nonnull), width = 0.15, size = 0.9, alpha = 0.7
        ) +
        facet_wrap(response_type ~ n, scales = "free_y") +
        scale_color_scico(palette = "berlin") +
        labs(
            x = NULL, y = glue("{which_method} score"),
            title = glue("{which_method} scores by feature: {which_dataset_id}")
        ) +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))
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
