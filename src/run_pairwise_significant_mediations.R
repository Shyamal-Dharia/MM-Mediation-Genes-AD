## ================================================================
## run_pairwise_significant_mediations.R
## Pairwise mediation for significant (y, mediator) pairs
## Comparisons:
##   1. 0 vs 1   (N vs A+P-)
##   2. 0 vs 2   (N vs A+P+)
##   3. 1 vs 2   (A+P- vs A+P+)
##
## Saves 3 CSV files with ACME/ADE/Total + EXTRA:
##   a-path (gene_bin -> mediator), b-path (mediator -> y | gene_bin),
##   c' (direct effect from outcome model), c (total model),
##   plus simple group means for mediator and outcome.
##
## Convention:
##   gene_bin = 0 => g1 (control)
##   gene_bin = 1 => g2 (treatment)
## ================================================================

library(mediation)
library(tidyverse)

## -----------------------------
## 1. Paths
## -----------------------------
data_path <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_ready_all_smri_eeg_fmri.csv"
sig_path  <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_grid_significant_acme.csv"

## Output CSVs
out_01 <- "pairwise_mediation_0_vs_1.csv"
out_02 <- "pairwise_mediation_0_vs_2.csv"
out_12 <- "pairwise_mediation_1_vs_2.csv"

## -----------------------------
## 2. Load data
## -----------------------------
dat <- read.csv(data_path, check.names = TRUE)
sig <- read.csv(sig_path, check.names = TRUE)

stopifnot(all(c("y", "mediator") %in% names(sig)))
stopifnot("gene" %in% names(dat))

if (is.factor(dat$gene)) dat$gene <- as.numeric(as.character(dat$gene))

## -----------------------------
## 3. Covariates
## -----------------------------
covars_raw <- c(
  "age", "sex", "education",
  "learning_deficits",
  "HSV_r", "BMI",
  "hypertension", "thyroid_diseases",
  "other_diseases", "dementia_history_parents"
)

covars <- covars_raw[covars_raw %in% names(dat)]

if (length(covars) > 0) {
  dat[covars] <- lapply(dat[covars], function(x) {
    if (is.character(x)) factor(x) else x
  })
}

## -----------------------------
## 4. Choose family of Y
## -----------------------------
choose_family_y <- function(x) {
  ux <- sort(unique(na.omit(x)))
  if (length(ux) == 2 && all(ux %in% c(0, 1))) {
    list(name = "binomial", fam = binomial())
  } else {
    list(name = "gaussian", fam = gaussian())
  }
}

## -----------------------------
## 5. Helpers: safe formula builder (no fallbacks, just deterministic)
## -----------------------------
make_formula <- function(lhs, rhs_terms) {
  stopifnot(is.character(lhs), length(lhs) == 1)
  stopifnot(is.character(rhs_terms))
  rhs_terms <- rhs_terms[nzchar(rhs_terms)]
  if (length(rhs_terms) == 0) stop("No RHS terms provided for formula.")
  as.formula(paste(lhs, "~", paste(rhs_terms, collapse = " + ")))
}

## -----------------------------
## 6. Function: run mediation for one comparison
## -----------------------------
run_pairwise <- function(dat, sig, g1, g2, outfile) {

  message("\n==============================")
  message("Running comparison: ", g1, " vs ", g2)
  message("==============================")

  df <- dat %>% filter(gene %in% c(g1, g2))
  if (nrow(df) < 30) stop("Too few subjects for comparison.")

  ## re-code gene → 0/1 for mediation
  df <- df %>% mutate(gene_bin = ifelse(gene == g1, 0, 1))

  results <- list()

  for (i in seq_len(nrow(sig))) {

    y_name <- sig$y[i]
    m_name <- sig$mediator[i]

    message("\n----- Pair ", i, ": Y=", y_name, " | M=", m_name, " -----")

    stopifnot(y_name %in% names(df))
    stopifnot(m_name %in% names(df))

    fam_info <- choose_family_y(df[[y_name]])
    fam_y    <- fam_info$fam

    vars_needed <- c("gene_bin", y_name, m_name, covars)
    df_sub <- df[complete.cases(df[, vars_needed, drop = FALSE]), vars_needed, drop = FALSE]

    if (nrow(df_sub) < 25) next
    if (var(df_sub$gene_bin) == 0) next
    if (var(df_sub[[m_name]]) == 0) next

    ## Outcome and mediator formulas
    f_out <- make_formula(
      lhs = y_name,
      rhs_terms = c("gene_bin", m_name, covars)
    )
    f_med <- make_formula(
      lhs = m_name,
      rhs_terms = c("gene_bin", covars)
    )

    out_model <- glm(f_out, data = df_sub, family = fam_y)
    med_model <- glm(f_med, data = df_sub, family = gaussian())

    ## ---- NEW: extract a-path (gene_bin -> mediator) ----
    smed <- summary(med_model)$coefficients
    stopifnot("gene_bin" %in% rownames(smed))
    a_hat <- smed["gene_bin", "Estimate"]
    a_se  <- smed["gene_bin", "Std. Error"]
    a_p   <- smed["gene_bin", "Pr(>|t|)"]

    ## ---- NEW: extract b-path (mediator -> y | gene_bin) and c' ----
    sout <- summary(out_model)$coefficients
    stopifnot(m_name %in% rownames(sout))
    stopifnot("gene_bin" %in% rownames(sout))

    b_hat <- sout[m_name, "Estimate"]
    b_se  <- sout[m_name, "Std. Error"]
    cprime_hat <- sout["gene_bin", "Estimate"]
    cprime_se  <- sout["gene_bin", "Std. Error"]

    if (fam_info$name == "binomial") {
      stopifnot("Pr(>|z|)" %in% colnames(sout))
      b_p      <- sout[m_name, "Pr(>|z|)"]
      cprime_p <- sout["gene_bin", "Pr(>|z|)"]
    } else {
      stopifnot("Pr(>|t|)" %in% colnames(sout))
      b_p      <- sout[m_name, "Pr(>|t|)"]
      cprime_p <- sout["gene_bin", "Pr(>|t|)"]
    }

    ## ---- NEW (optional): total model coefficient c (gene_bin -> y) ----
    f_total <- make_formula(
      lhs = y_name,
      rhs_terms = c("gene_bin", covars)
    )
    total_model <- glm(f_total, data = df_sub, family = fam_y)
    stot <- summary(total_model)$coefficients
    stopifnot("gene_bin" %in% rownames(stot))
    c_hat <- stot["gene_bin", "Estimate"]
    c_se  <- stot["gene_bin", "Std. Error"]
    if (fam_info$name == "binomial") {
      stopifnot("Pr(>|z|)" %in% colnames(stot))
      c_p <- stot["gene_bin", "Pr(>|z|)"]
    } else {
      stopifnot("Pr(>|t|)" %in% colnames(stot))
      c_p <- stot["gene_bin", "Pr(>|t|)"]
    }

    ## ---- NEW (optional): simple observed means by group ----
    m_mean_g1 <- mean(df_sub[[m_name]][df_sub$gene_bin == 0])
    m_mean_g2 <- mean(df_sub[[m_name]][df_sub$gene_bin == 1])
    y_mean_g1 <- mean(df_sub[[y_name]][df_sub$gene_bin == 0])
    y_mean_g2 <- mean(df_sub[[y_name]][df_sub$gene_bin == 1])

    ## Mediation
    med_fit <- mediate(
      model.m       = med_model,
      model.y       = out_model,
      treat         = "gene_bin",
      mediator      = m_name,
      treat.value   = 1,
      control.value = 0,
      sims          = 10000
    )

    sumfit <- summary(med_fit)

    results[[i]] <- data.frame(
      y = y_name,
      mediator = m_name,

      ## ACME/ADE/Total from mediation package
      acme = sumfit$d0,
      acme_p = sumfit$d0.p,
      acme_ci_low = sumfit$d0.ci[1],
      acme_ci_high = sumfit$d0.ci[2],

      ade = sumfit$z0,
      ade_p = sumfit$z0.p,
      ade_ci_low = sumfit$z0.ci[1],
      ade_ci_high = sumfit$z0.ci[2],

      total = sumfit$tau.coef,
      total_p = sumfit$tau.p,
      total_ci_low = sumfit$tau.ci[1],
      total_ci_high = sumfit$tau.ci[2],

      prop_med = sumfit$n0,

      ## NEW: path coefficients + means for interpretation
      a_hat = a_hat, a_se = a_se, a_p = a_p,
      b_hat = b_hat, b_se = b_se, b_p = b_p,
      cprime_hat = cprime_hat, cprime_se = cprime_se, cprime_p = cprime_p,
      c_hat = c_hat, c_se = c_se, c_p = c_p,
      m_mean_g1 = m_mean_g1, m_mean_g2 = m_mean_g2,
      y_mean_g1 = y_mean_g1, y_mean_g2 = y_mean_g2
    )
  }

  results_df <- bind_rows(results)
  write.csv(results_df, outfile, row.names = FALSE)
  message("Saved results to: ", outfile)

  return(results_df)
}

## -----------------------------
## 7. Run 3 comparisons
## -----------------------------
res_01 <- run_pairwise(dat, sig, g1 = 0, g2 = 1, outfile = out_01)
res_02 <- run_pairwise(dat, sig, g1 = 0, g2 = 2, outfile = out_02)
res_12 <- run_pairwise(dat, sig, g1 = 1, g2 = 2, outfile = out_12)

message("\nAll pairwise analyses complete.")
