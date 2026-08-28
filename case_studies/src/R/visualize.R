#' Visualize the attribution results

library(tidyverse)
library(fs)

#' Color anchors for the case-study figures.
#'
#' Drawn from two inspiration images: a deep-blue motif painting and a warm
#' grayscale op-art print. \code{diverging} runs negative -> zero -> positive
#' with warm ivory as the visual zero.
axiom_palette <- list(
    ink = "#2A2724",
    bg = "#FFFFFF",
    grid = "#E2DBD4",
    accent = "#066AE5",
    diverging = rev(c("#0433DC", "#049fdc", "#E2DBD4", "#fb315d", "#dc0433"))
)

#' A minimal, warm-neutral theme for the attribution figures.
#'
#' @param base_size Numeric. Base font size.
#' @return A ggplot2 theme object.
theme_axiomatic <- function(base_size = 10) {
    theme_classic(base_size = base_size) +
        theme(
            text = element_text(color = axiom_palette$ink),
            plot.title = element_text(hjust = 0, face = "plain", size = rel(1.1)),
            plot.background = element_rect(fill = axiom_palette$bg, color = NA),
            panel.background = element_rect(fill = axiom_palette$bg, color = NA),
            legend.background = element_rect(fill = axiom_palette$bg, color = NA),
            panel.border = element_blank(),
            panel.grid.major = element_line(color = axiom_palette$grid, linewidth = 0.3),
            panel.grid.minor = element_blank(),
            axis.text = element_text(color = axiom_palette$ink),
            axis.ticks = element_blank(),
            legend.position = "right",
            legend.key.size = unit(0.8, "cm")
        )
}

#' Diverging fill scale for SHAP / attribution values.
#'
#' @param limits Numeric length-2 vector or \code{NULL}. Fill limits.
#' @param name Character. Legend title.
#' @return A ggplot2 fill scale.
scale_fill_shap <- function(limits = NULL, name = "SHAP") {
    scale_fill_gradientn(colours = axiom_palette$diverging, limits = limits, name = name)
}

#' Load attributions and sample metadata from a case study results directory.
#'
#' @param base_dir Character. Path to the case study root (containing
#'   \code{results/}).
#' @return Named list with tibbles \code{shap}, \code{minshap}, \code{meta}.
load_attributions <- function(base_dir) {
    res <- path(base_dir, "results")
    list(
        shap = read_csv(path(res, "shap_attributions.csv"), show_col_types = FALSE),
        minshap = read_csv(path(res, "minshap_attributions.csv"), show_col_types = FALSE),
        meta = read_csv(path(res, "patient_meta.csv"), show_col_types = FALSE)
    )
}


#' Bar charts of the top-k attributed features
#'
#' @param row_idx Integer. Row in \code{attr_df} to visualize.
#' @param attr_df Data frame of attributions (samples x features).
#' @param method_label Character label for the x-axis, e.g. \code{"SHAP"}.
#' @param top_n Integer. Number of features to display.
#' @return A ggplot object.
#' Barplot of a patient's top attributions.
#'
#' For minSHAP, the sign actually matters. For SHAP, we consider only
#' magnitude.
attribution_barplot <- function(row_idx, attr_df, method_label, top_n = 10, by_magnitude = TRUE) {
    vals <- as.numeric(attr_df[row_idx, ])
    names(vals) <- names(attr_df)
    rank_key <- if (by_magnitude) abs(vals) else vals
    top_idx <- head(order(rank_key, decreasing = TRUE), top_n)
    tibble(
        feature = names(vals)[top_idx],
        value = vals[top_idx]
    ) |>
        mutate(feature = fct_reorder(feature, if (by_magnitude) abs(value) else value)) |>
        ggplot(aes(value, feature)) +
        geom_col(fill = axiom_palette$ink) +
        labs(x = method_label, y = NULL) +
        theme_axiomatic(base_size = 10)
}
