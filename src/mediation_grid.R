## ======================================================================
## Bulk mediation grid: gene -> med_* -> y_*
## Uses mediation::mediate for ACME / ADE / Total effect
## ======================================================================

## Install once if needed:
## install.packages("mediation")
## install.packages("tidyverse")

library(mediation)
library(tidyverse)

## -----------------------------
## 1. Load data
## -----------------------------

data_path <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_ready_all_smri_eeg_fmri.csv"

dat <- read.csv(data_path, check.names = TRUE)

## Make sure exposure exists
if (!"gene" %in% names(dat)) {
  stop("Expected column 'gene' not found in data. Create it from GROUP before running.")
}

## Treat gene as numeric (0/1/2 etc.)
if (is.factor(dat$gene)) {
  dat$gene <- as.numeric(as.character(dat$gene))
}

## -----------------------------
## 2. Define variables
## -----------------------------

treat_var <- "gene"

med_vars <- grep("^med_", names(dat), value = TRUE)
y_vars   <- grep("^y_",   names(dat), value = TRUE)

if (length(med_vars) == 0) stop("No columns with prefix 'med_' found.")
if (length(y_vars)   == 0) stop("No columns with prefix 'y_' found.")

## Covariates – EDIT THIS to what you want
covars_raw <- c("age", "sex", "education", "learning_deficits", "HSV_r", "BMI", "hypertension", "thyroid_diseases", "other_diseases", "dementia_history_parents") #"HSV_r", "BMI", "hypertension", "thyroid_diseases", "other_diseases", "dementia_history_parents" )

## Keep only covars that actually exist in the data
covars <- covars_raw[covars_raw %in% names(dat)]
if (length(covars) > 0) {
  message("Using covariates: ", paste(covars, collapse = ", "))
} else {
  message("No covariates found in data. Running mediation without covariates.")
}

## Convert character covariates to factors
if (length(covars) > 0) {
  dat[covars] <- lapply(dat[covars], function(x) {
    if (is.character(x)) factor(x) else x
  })
}

## -----------------------------
## 3. Helper: choose family for Y
## -----------------------------

choose_family_y <- function(x) {
  ux <- sort(unique(na.omit(x)))
  if (length(ux) == 2 && all(ux %in% c(0, 1))) {
    return(list(name = "binomial", fam = binomial()))
  } else {
    return(list(name = "gaussian", fam = gaussian()))
  }
}

## -----------------------------
## 4. Main grid loop
## -----------------------------

results <- list()
idx <- 1L

for (y in y_vars) {
  cat("\n==============================\n")
  cat("Outcome:", y, "\n")
  cat("==============================\n")

  fam_info <- choose_family_y(dat[[y]])
  fam_y    <- fam_info$fam
  fam_name <- fam_info$name

  for (m in med_vars) {
    cat("  Mediator:", m, " ... ")

    vars_needed <- c(treat_var, y, m, covars)
    vars_needed <- vars_needed[vars_needed %in% names(dat)]

    ## Drop rows with any NA in the relevant vars
    complete_idx <- complete.cases(dat[, vars_needed])
    df_sub <- dat[complete_idx, vars_needed, drop = FALSE]

    ## Basic sanity filters
    if (nrow(df_sub) < 40) {
      cat("skip (n <", 40, ")\n")
      next
    }
    if (var(df_sub[[treat_var]], na.rm = TRUE) == 0) {
      cat("skip (no variation in gene)\n")
      next
    }
    if (var(df_sub[[m]], na.rm = TRUE) == 0) {
      cat("skip (mediator has zero variance)\n")
      next
    }

    ## Formulas
    rhs_out <- c(treat_var, m, covars)
    f_out <- as.formula(
      paste(y, "~", paste(rhs_out, collapse = " + "))
    )

    rhs_med <- c(treat_var, covars)
    f_med <- as.formula(
      paste(m, "~", paste(rhs_med, collapse = " + "))
    )

    ## Fit models
    out_model <- tryCatch(
      glm(f_out, data = df_sub, family = fam_y),
      error = function(e) NULL
    )
    if (is.null(out_model)) {
      cat("skip (glm outcome failed)\n")
      next
    }

    med_model <- tryCatch(
      glm(f_med, data = df_sub, family = gaussian()),
      error = function(e) NULL
    )
    if (is.null(med_model)) {
      cat("skip (glm mediator failed)\n")
      next
    }

    ## Define treatment / control values for gene
    tmin <- min(df_sub[[treat_var]], na.rm = TRUE)
    tmax <- max(df_sub[[treat_var]], na.rm = TRUE)
    if (isTRUE(all.equal(tmin, tmax))) {
      cat("skip (gene constant)\n")
      next
    }

    ## Run causal mediation
    med_fit <- tryCatch(
      mediate(
        model.m      = med_model,
        model.y      = out_model,
        treat        = treat_var,
        mediator     = m,
        treat.value  = tmax,
        control.value = tmin,
        sims         = 10000
      ),
      error = function(e) {
        cat("skip (mediate error:", e$message, ")\n")
        NULL
      }
    )

    if (is.null(med_fit)) next

    ## Collect main quantities
    res_row <- data.frame(
      y           = y,
      mediator    = m,
      n           = med_fit$nobs,
      family_y    = fam_name,

      acme        = med_fit$d.avg,
      acme_ci_low = med_fit$d.avg.ci[1],
      acme_ci_high= med_fit$d.avg.ci[2],
      acme_p      = med_fit$d.avg.p,

      ade         = med_fit$z.avg,
      ade_ci_low  = med_fit$z.avg.ci[1],
      ade_ci_high = med_fit$z.avg.ci[2],
      ade_p       = med_fit$z.avg.p,

      total_effect    = med_fit$tau.coef,
      total_ci_low    = med_fit$tau.ci[1],
      total_ci_high   = med_fit$tau.ci[2],
      total_p         = med_fit$tau.p,

      prop_med        = med_fit$n.avg,
      prop_med_ci_low = med_fit$n.avg.ci[1],
      prop_med_ci_high= med_fit$n.avg.ci[2],
      prop_med_p      = med_fit$n.avg.p,

      stringsAsFactors = FALSE
    )

    results[[idx]] <- res_row
    idx <- idx + 1L

    cat("done (ACME p =", sprintf("%.3g", res_row$acme_p), ")\n")
  }
}

if (length(results) == 0) {
  stop("No successful mediation fits. Check variable names / families / covariates.")
}

all_res <- bind_rows(results)

## -----------------------------
## 5. FDR and save
## -----------------------------

all_res$acme_p_fdr <- p.adjust(all_res$acme_p, method = "fdr")

sig_res <- all_res %>%
  filter(acme_p_fdr < 0.05)

out_all <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_grid_all_pairs.csv"
out_sig <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_grid_significant_acme.csv"

write.csv(all_res, out_all, row.names = FALSE)
write.csv(sig_res, out_sig, row.names = FALSE)

cat("\n=====================================================\n")
cat("Saved ALL results to:        ", out_all, "\n")
cat("Saved SIGNIFICANT (FDR<0.05) to:", out_sig, "\n")
cat("Number of significant (ACME FDR<0.05):", nrow(sig_res), "\n")
cat("Done.\n")
cat("=====================================================\n")
