# Mock "SummarizedExperiment" for testing src/intake/R/export_se_tables_v2.R without Bioconductor: S4 objects with the same slot layout idea.
# Main experiment: 5 gene rows x 3 cells; an alternative experiment ("guides") has 8 guide names (the DLD-1 trap: a NAMES vector of the wrong length).
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
two_candidates <- length(args) > 1 && args[2] == "ambiguous"
setClass("MockSeed", representation(dim = "integer", filepath = "character"))
setClass("MockPart", representation(NAMES = "character"))
setClass("MockGRL", representation(partitioning = "MockPart"))
setClass("MockDF", representation(listData = "list", rownames = "character"))
setClass("MockAlt", representation(NAMES = "character", assays = "list"))
setClass("MockSCE", representation(colData = "MockDF", assays = "list", rowRanges = "MockGRL", altExps = "list", NAMES = "character"))
main_names <- paste0("GENE", 1:5)
se <- new("MockSCE",
          colData = new("MockDF", listData = list(assignment_v2 = c("NTC", "STAT1", "JUN"), Sample = factor(c("s1", "s1", "s2"))), rownames = c("AAA-1", "AAA-2", "BBB-1")),
          assays = list(new("MockSeed", dim = c(5L, 3L), filepath = "x_assays.h5")),
          rowRanges = new("MockGRL", partitioning = new("MockPart", NAMES = main_names)),
          altExps = list(guides = new("MockAlt", NAMES = paste0("GUIDE_P1_", 1:8), assays = list(new("MockSeed", dim = c(8L, 3L), filepath = "g.h5")))),
          NAMES = if (two_candidates) rev(main_names) else character(0))
saveRDS(se, out)
