# Visualization helpers for sanity checks notebook

library(glue)

#' Load dataset from CSV
#' @param name dataset name
#' @param n sample size
#' @param data_dir directory containing data (default: "data")
read_dataset <- function(name, n, data_dir = "data") {
    read_csv(file.path(data_dir, glue("{name}_{n}.csv")), show_col_types = FALSE)
}

#' Marginal correlation (Pearson)
#' @param data data frame
#' @param feature feature name
#' @param response response variable name (default: "y")
marginal_cor <- function(data, feature, response = "y") {
    cor(data[[feature]], data[[response]], use = "complete.obs")
}

#' Partial correlation via residualization
#' @param data data frame
#' @param feature feature name
#' @param response response variable name (default: "y")
partial_cor <- function(data, feature, response = "y") {
    other_features <- setdiff(names(data), c(feature, response))

    # Residualize feature
    lm_feat <- lm(as.formula(paste(feature, "~", paste(other_features, collapse = "+"))), data = data)
    resid_feat <- residuals(lm_feat)

    # Residualize response
    lm_resp <- lm(as.formula(paste(response, "~", paste(other_features, collapse = "+"))), data = data)
    resid_resp <- residuals(lm_resp)
    cor(resid_feat, resid_resp, use = "complete.obs")
}

#' Read same dataset across multiple sample sizes, tagging each row with n
#' @param name dataset name
#' @param ns vector of sample sizes to compare (default c(50, 500, 5000))
#' @param data_dir directory containing data
read_dataset_across_n <- function(name, ns = c(50, 500, 5000), data_dir = "data") {
    map_dfr(ns, ~ read_dataset(name, .x, data_dir) |> mutate(n = .x))
}

#' Marginal and partial correlation table for a set of features
#' @param data data frame
#' @param features character vector of feature names to check
#' @param response response variable name (default "y")
correlation_table <- function(data, features, response = "y") {
    data.frame(
        Feature = features,
        Marginal = map_dbl(features, ~ marginal_cor(data, .x, response)),
        Partial = map_dbl(features, ~ partial_cor(data, .x, response))
    )
}

#' Faceted binned boxplot across sample sizes, with features classified into
#' labeled groups (e.g. signal/null) for coloring
#' @param name dataset name
#' @param features character vector of features to include
#' @param classify function (or purrr-style formula) mapping feature name -> label
#' @param fill_values named vector of label -> color for scale_fill_manual
#' @param ns sample sizes to compare (default c(50, 500, 5000))
#' @param data_dir directory containing data
plot_facet_across_n <- function(name, features, classify, fill_values,
                                ns = c(50, 500, 5000), data_dir = "data") {
    classify <- as_mapper(classify)
    dat_all <- read_dataset_across_n(name, ns, data_dir) |>
        pivot_longer(cols = all_of(features), names_to = "feature", values_to = "value") |>
        mutate(
            feature_type = classify(feature),
            value_bin = cut(value, breaks = 8)
        )

    plot_facet_binned(dat_all, fill_var = "feature_type") +
        scale_fill_manual(values = fill_values)
}

#' Quantile binning helper
#' @param x numeric vector
#' @param n_quantiles number of quantiles to create
quantile_bin <- function(x, n_quantiles = 3) {
    n_unique <- length(unique(x))
    n_q <- min(n_quantiles, n_unique)
    if (n_q < 2) {
        return(factor(rep("Q1", length(x))))
    }
    breaks <- unique(quantile(x, probs = seq(0, 1, length.out = n_q + 1)))
    cut(x,
        breaks = breaks, labels = paste0("Q", 1:(length(breaks) - 1)),
        include.lowest = TRUE
    )
}

#' Plot plain scatter (no binning)
#' @param data data frame
#' @param x_var name of x variable (character)
#' @param y_var name of y variable (default "y")
#' @param title plot title
#' @param color point/smooth color
plot_scatter_plain <- function(data, x_var, y_var = "y", title = "", color = "#1f77b4") {
    ggplot(data, aes(x = .data[[x_var]], y = .data[[y_var]])) +
        geom_point(
            alpha = 0.4, size = 1.8, color = color,
            position = position_jitter(height = 0.05, width = 0)
        ) +
        geom_smooth(
            method = "loess", span = 0.8, color = "#333333",
            fill = NA, se = FALSE, linewidth = 0.5
        ) +
        labs(title = title, x = x_var, y = y_var) +
        theme(legend.position = "none")
}

#' Plot binned scatter with per-bin smooths
#' @param data data frame
#' @param x_var name of x variable (character)
#' @param bin_var name of binning variable (character)
#' @param y_var name of y variable (default "y")
#' @param title plot title
#' @param bin_name name for legend
#' @param palette_quantiles color palette for bins (length 3)
plot_scatter_binned <- function(data, x_var, bin_var, y_var = "y",
                                title = "", bin_name = "Bin",
                                palette_quantiles = c("#d73027", "#fee090", "#1a9850")) {
    ggplot(data, aes(
        x = .data[[x_var]], y = .data[[y_var]],
        color = .data[[bin_var]]
    )) +
        geom_point(
            alpha = 0.5, size = 1.8,
            position = position_jitter(height = 0.05, width = 0)
        ) +
        geom_smooth(
            method = "loess", span = 0.8,
            fill = NA, se = FALSE, linewidth = 0.5
        ) +
        scale_color_manual(
            values = setNames(
                palette_quantiles,
                c("Q1", "Q2", "Q3")
            ),
            name = bin_name
        ) +
        labs(title = title, x = x_var, y = y_var) +
        theme(legend.position = "right")
}

#' Three-panel scatter plot using patchwork
#' Always shows: x1 vs y (plain) | x1 vs y binned by bin_feat | x3 vs y (plain)
#' @param data data frame
#' @param x_plain1 first plain x variable
#' @param x_binned x variable for x-axis of middle panel
#' @param bin_feat feature to bin by (for coloring middle panel)
#' @param x_plain2 second plain x variable
#' @param palette_quantiles color palette
compose_scatter_panels <- function(data, x_plain1, x_binned, bin_feat,
                                   x_plain2,
                                   palette_quantiles = c("#d73027", "#fee090", "#1a9850")) {
    # Create binned variable from bin_feat
    bin_var <- glue("{bin_feat}_quantile")
    data_binned <- data |>
        mutate(!!bin_var := quantile_bin(.data[[bin_feat]], 3))

    p1 <- plot_scatter_plain(data, x_plain1, title = glue("{x_plain1} vs y"))
    p2 <- plot_scatter_binned(data_binned, x_binned, bin_var,
        title = glue("{x_plain1} vs y | {bin_feat} (binned)"),
        bin_name = glue("{bin_feat} quantile"),
        palette_quantiles = palette_quantiles
    )
    p3 <- plot_scatter_plain(data, x_plain2, title = glue("{x_plain2} vs y"))

    # Patchwork composition with shared y-axis
    p1 + p2 + p3 +
        plot_layout(ncol = 3, widths = c(1, 1.1, 1)) &
        theme(axis.title.y = element_blank())
}

#' Create faceted binned boxplot across sample sizes
#' @param data data frame (already has features as rows via pivot_longer)
#' @param fill_var variable to use for fill color
plot_facet_binned <- function(data, fill_var = "feature_type") {
    ggplot(data, aes(
        x = .data$value_bin, y = .data$y,
        fill = .data[[fill_var]]
    )) +
        geom_boxplot(alpha = 0.6, width = 0.7, outlier.size = 1) +
        facet_grid(.data$feature ~ .data$n, scales = "free_x") +
        labs(
            title = "Binned marginal effects across sample sizes",
            x = "Feature (binned)", y = "y"
        ) +
        theme(
            axis.text.x = element_blank(),
            legend.position = "top"
        )
}
