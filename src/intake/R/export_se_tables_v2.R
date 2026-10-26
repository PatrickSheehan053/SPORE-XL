# export_se_tables_v2.R: export the cell table and the GENE table of an HDF5-backed SummarizedExperiment / SingleCellExperiment .rds
# (first case: DLD-1 GSE337988, `*_se.rds` + `*_assays.h5`), without needing Bioconductor: S4 objects are walked through attributes().
#
# Why v2: export_se_metadata_v1.R took the first character vector called NAMES it found. For DLD-1 that was a vector of 23,480 GUIDE names
# (A1CF_P1_1, ...) while the assay in the .h5 has 36,601 gene rows, so the gene axis did not match. This version
#   1. reads the assay dimensions from the HDF5 seed (attribute dim, found by walking the object),
#   2. lists EVERY character vector it can find (path, class, length, first values) in candidates_<name>.tsv,
#   3. writes genes.tsv from the unique candidate whose length equals the number of assay rows (or columns),
#      and STOPS with the candidate table when there is none or more than one (then pick the path yourself: --gene-path).
# Usage:  Rscript export_se_tables_v2.R <se.rds> <out_dir> [--gene-path "a/b/NAMES"]
# Output: <out_dir>/cells.tsv.gz  (cell_index, barcode, all colData columns)
#         <out_dir>/genes.tsv     (gene_index, gene_id)             only when the gene axis could be decided
#         <out_dir>/candidates.tsv, <out_dir>/dims.tsv, <out_dir>/export_se_tables_v2.log
# Writes only new files into a new or empty out_dir; refuses to overwrite.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript export_se_tables_v2.R <se.rds> <out_dir> [--gene-path path]")
rds_path <- args[1]
out_dir <- args[2]
gene_path <- if ("--gene-path" %in% args) args[which(args == "--gene-path") + 1] else NA_character_
if (dir.exists(out_dir) && length(list.files(out_dir)) > 0) stop(paste("refusing to write into a non-empty folder:", out_dir))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_path <- file.path(out_dir, "export_se_tables_v2.log")
say <- function(...) { text <- paste0(...); cat(text, "\n"); cat(text, "\n", file = log_path, append = TRUE) }

# ---- the walk: every node is a list / S4 object; children = list elements + attributes (S4 slots are attributes)
walk <- function(node, path = "", depth = 0, max_depth = 14, found = new.env()) {
  if (is.null(found$chars)) { found$chars <- list(); found$dims <- list(); found$visited <- 0 }
  found$visited <- found$visited + 1
  if (depth > max_depth) return(found)
  if (is.character(node) && length(node) > 1) {
    found$chars[[length(found$chars) + 1]] <- list(path = path, n = length(node), head = paste(head(node, 3), collapse = ","), value = node)
  }
  if (!is.null(attr(node, "dim")) && is.numeric(attr(node, "dim")) && length(attr(node, "dim")) == 2 && path != "") {
    found$dims[[length(found$dims) + 1]] <- list(path = path, d1 = attr(node, "dim")[1], d2 = attr(node, "dim")[2])
  }
  children <- list()
  if (is.list(node) && !is.data.frame(node)) {
    nm <- names(node)
    for (i in seq_along(node)) children[[paste0(path, "/", if (!is.null(nm) && nzchar(nm[i])) nm[i] else paste0("[", i, "]"))]] <- node[[i]]
  }
  atts <- attributes(node)
  for (a in names(atts)) {
    if (a %in% c("class", "names", "row.names", "dim")) next
    children[[paste0(path, "@", a)]] <- atts[[a]]
  }
  # S4 "dim" slots stored as plain integer vectors named dim / dimnames of seeds
  for (child_path in names(children)) walk(children[[child_path]], child_path, depth + 1, max_depth, found)
  found
}

say("reading ", rds_path)
se <- readRDS(rds_path)
say("class: ", paste(class(se), collapse = ","))
found <- walk(se, path = "se")
say("visited ", found$visited, " nodes; ", length(found$chars), " character vectors; ", length(found$dims), " dim attributes")

chars <- do.call(rbind, lapply(found$chars, function(item) data.frame(path = item$path, length = item$n, head = item$head, stringsAsFactors = FALSE)))
if (is.null(chars)) chars <- data.frame(path = character(), length = integer(), head = character())
write.table(chars, file.path(out_dir, "candidates.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
dims <- do.call(rbind, lapply(found$dims, function(item) data.frame(path = item$path, d1 = item$d1, d2 = item$d2)))
if (is.null(dims)) dims <- data.frame(path = character(), d1 = numeric(), d2 = numeric())
write.table(dims, file.path(out_dir, "dims.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# ---- cells: colData is the DataFrame in the 'colData' slot; its columns live in listData
col_data <- attr(se, "colData")
cell_table <- NULL
if (!is.null(col_data)) {
  list_data <- attr(col_data, "listData")
  if (is.list(list_data) && length(list_data) > 0) {
    cell_table <- as.data.frame(lapply(list_data, function(column) if (is.factor(column)) as.character(column) else column), stringsAsFactors = FALSE, check.names = FALSE)
    cell_names <- attr(attr(col_data, "rownames"), "x")
    if (is.null(cell_names)) cell_names <- attr(col_data, "rownames")
    if (is.character(cell_names) && length(cell_names) == nrow(cell_table)) cell_table <- cbind(barcode = cell_names, cell_table)
    cell_table <- cbind(cell_index = seq_len(nrow(cell_table)) - 1, cell_table)
  }
}
if (is.null(cell_table)) {
  say("could not read colData through attributes(); see candidates.tsv")
} else {
  gz <- gzfile(file.path(out_dir, "cells.tsv.gz"), "w")
  write.table(cell_table, gz, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz)
  say("cells.tsv.gz: ", nrow(cell_table), " rows, ", ncol(cell_table), " columns")
}

# ---- genes: the character vector whose length equals the number of assay rows (or columns)
main_dims <- dims[!grepl("altExp", dims$path), ]   # alternative experiments (guide library, ADT, ...) have their own row axis: not the gene axis
assay_dims <- if (nrow(main_dims) > 0) unique(c(main_dims$d1, main_dims$d2)) else numeric()
say("dim attributes found: ", paste(unique(paste0(dims$d1, "x", dims$d2)), collapse = "; "), "; main experiment: ", paste(unique(paste0(main_dims$d1, "x", main_dims$d2)), collapse = "; "))
choose_vector <- function() {
  if (!is.na(gene_path)) {
    hit <- chars[chars$path == gene_path, ]
    if (nrow(hit) != 1) stop(paste("--gene-path not found among the candidates:", gene_path))
    return(hit$path)
  }
  big <- if (!is.null(cell_table)) nrow(cell_table) else NA
  wanted <- setdiff(assay_dims, big)   # the gene axis is the assay dimension that is not the cell count
  hits <- chars[chars$length %in% wanted & !grepl("altExp", chars$path), ]
  if (nrow(hits) == 0) { say("NO character vector has a length equal to the gene axis (", paste(wanted, collapse = " or "), "). Look at candidates.tsv."); return(NULL) }
  if (length(unique(hits$path)) > 1) {
    say("SEVERAL character vectors have the gene-axis length ", paste(wanted, collapse = " or "), ":")
    for (i in seq_len(nrow(hits))) say("   ", hits$path[i], "  (", hits$length[i], ")  ", hits$head[i])
    say("Pick one and run again with --gene-path \"<path>\" into a NEW folder.")
    return(NULL)
  }
  hits$path[1]
}
selected <- choose_vector()
if (!is.null(selected)) {
  vector <- NULL
  for (item in found$chars) if (identical(item$path, selected)) vector <- item$value
  genes <- data.frame(gene_index = seq_along(vector) - 1, gene_id = as.character(vector), stringsAsFactors = FALSE)
  write.table(genes, file.path(out_dir, "genes.tsv"), sep = "	", quote = FALSE, row.names = FALSE)
  say("genes.tsv: ", nrow(genes), " rows from ", selected, " (first: ", paste(head(genes$gene_id, 3), collapse = ", "), ")")
} else {
  say("genes.tsv NOT written")
}
say("done")
