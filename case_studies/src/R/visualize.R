#' Visualize the attribution results

library(tidyverse)
library(ggthemes)
library(fs)

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
attribution_barplot <- function(row_idx, attr_df, method_label, top_n = 10) {
    vals <- as.numeric(attr_df[row_idx, ])
    names(vals) <- names(attr_df)
    top_idx <- head(order(abs(vals), decreasing = TRUE), top_n)
    tibble(
        feature = names(vals)[top_idx],
        value = vals[top_idx]
    ) |>
        mutate(feature = fct_reorder(feature, abs(value))) |>
        ggplot(aes(value, feature, fill = value > 0)) +
        geom_col() +
        scale_fill_manual(
            values = c(`TRUE` = "#31a354", `FALSE` = "#e34a33"),
            guide = "none"
        ) +
        labs(x = method_label, y = NULL) +
        theme_economist(base_size = 8)
}
