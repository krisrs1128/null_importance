#!/usr/bin/env Rscript
#
# This script downloads the TCGA data needed for the multi-omics disease
# prediction case study.
#
#    Rscript download_tcga.R

library(TCGAbiolinks)
library(SummarizedExperiment)
library(stringr)
library(purrr)
library(fs)
library(glue)

OUT_DIR <- path("data", "raw")
dir_create(OUT_DIR, recurse = TRUE)

# Truncate barcodes to patient level (TCGA-XX-XXXX)
shorten <- \(x) substr(x, 1, 12)

# Remove duplicate columns by name
dedup_cols <- \(mat) mat[, !duplicated(colnames(mat)), drop = FALSE]

# Save matrix as TSV and log dimensions
save_tsv <- function(mat, fname) {
    path <- OUT_DIR / fname
    write.table(mat, path, sep = "\t", quote = FALSE, col.names = NA)
    message(glue("  -> {fname}  ({nrow(mat)} features × {ncol(mat)} samples)"))
    invisible(path)
}

already_done <- function(fname) {
    p <- OUT_DIR / fname
    if (file_exists(p)) {
        message(glue("  Skipping (already saved): {fname}"))
        return(TRUE)
    }
    FALSE
}


# mRNA expression
message("=== mRNA (STAR FPKM) ===")
if (!already_done("TCGA.BRCA.sampleMap_HiSeqV2.tsv")) {
    mrna_q <- GDCquery(
        project = "TCGA-BRCA",
        data.category = "Transcriptome Profiling",
        data.type = "Gene Expression Quantification",
        workflow.type = "STAR - Counts",
        sample.type = "Primary Tumor"
    )
    GDCdownload(mrna_q, method = "api", files.per.chunk = 50)
    mrna_se <- GDCprepare(mrna_q)

    mrna_mat <- log1p(assay(mrna_se, "fpkm_unstrand"))
    rownames(mrna_mat) <- rowData(mrna_se)$gene_name
    mrna_mat <- mrna_mat[!duplicated(rownames(mrna_mat)), ]
    colnames(mrna_mat) <- shorten(colnames(mrna_mat))
    mrna_mat <- dedup_cols(mrna_mat)
    save_tsv(mrna_mat, "TCGA.BRCA.sampleMap_HiSeqV2.tsv")
}


# miRNA expression
message("=== miRNA ===")
if (!already_done("TCGA.BRCA.sampleMap_miRNA_HiSeq_gene.tsv")) {
    mirna_q <- GDCquery(
        project = "TCGA-BRCA",
        data.category = "Transcriptome Profiling",
        data.type = "miRNA Expression Quantification",
        sample.type = "Primary Tumor"
    )
    GDCdownload(mirna_q, method = "api", files.per.chunk = 50)
    mirna_data <- GDCprepare(mirna_q)

    rpm_cols <- grep("reads_per_million_miRNA_mapped", names(mirna_data), value = TRUE)
    mirna_mat <- as.matrix(mirna_data[, rpm_cols])
    rownames(mirna_mat) <- mirna_data[["miRNA_ID"]]
    colnames(mirna_mat) <- sub("\\.reads_per_million_miRNA_mapped$", "", rpm_cols)

    # preprocess
    mirna_mat <- log1p(mirna_mat)
    colnames(mirna_mat) <- shorten(colnames(mirna_mat))
    mirna_mat <- dedup_cols(mirna_mat)
    save_tsv(mirna_mat, "TCGA.BRCA.sampleMap_miRNA_HiSeq_gene.tsv")
}


# Protein expression (RPPA)
message("=== Protein RPPA ===")
if (!already_done("TCGA.BRCA.sampleMap_RPPA_RBN.tsv")) {
    protein_q <- GDCquery(
        project = "TCGA-BRCA",
        data.category = "Proteome Profiling",
        data.type = "Protein Expression Quantification",
        sample.type = "Primary Tumor"
    )
    GDCdownload(protein_q, method = "api", files.per.chunk = 50)
    protein_data <- GDCprepare(protein_q)

    num_cols <- map_lgl(protein_data, is.numeric) %>% which()
    protein_mat <- as.matrix(protein_data[, num_cols])
    rownames(protein_mat) <- protein_data[[1]]
    colnames(protein_mat) <- shorten(colnames(protein_mat))
    protein_mat <- dedup_cols(protein_mat)
    save_tsv(protein_mat, "TCGA.BRCA.sampleMap_RPPA_RBN.tsv")
}


# Clinical data
message("=== Clinical ===")
if (!already_done("TCGA.BRCA.sampleMap_BRCA_clinicalMatrix.tsv")) {
    clin <- GDCquery_clinic("TCGA-BRCA", "clinical")

    htype <- ifelse(
        str_detect(clin$primary_diagnosis, regex("lobular", ignore_case = TRUE)) &
            !str_detect(clin$primary_diagnosis, regex("in situ", ignore_case = TRUE)),
        "Infiltrating Lobular Carcinoma",
        ifelse(
            str_detect(clin$primary_diagnosis, regex("duct", ignore_case = TRUE)) &
                !str_detect(clin$primary_diagnosis, regex("in situ", ignore_case = TRUE)),
            "Infiltrating Ductal Carcinoma",
            NA_character_
        )
    )

    clin_out <- data.frame(
        row.names = clin$submitter_id,
        histological_type = htype,
        age_at_initial_pathologic_diagnosis = round(clin$age_at_diagnosis / 365.25),
        gender = clin$gender,
        race = clin$race,
        stringsAsFactors = FALSE
    )
    save_tsv(clin_out, "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix.tsv")
}

message("\nAll done. Run pipeline.py to train models.")
