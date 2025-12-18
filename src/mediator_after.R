## run_significant_mediations.R
## Re-run mediation for each significant (y, mediator) pair

library(mediation)
library(tidyverse)

## -----------------------------
## 1. Paths
## -----------------------------
data_path <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_ready_all_smri_eeg_fmri.csv"
sig_path  <- "/data/s.dharia-ra/PEARL/MM-Mediation-Genes-AD/mediation_grid_significant_acme.csv"
## If your file is named differently (e.g., significant_acme_results_uncorrected.csv),
## change sig_path accordingly.

## -----------------------------
## 2. Load data
## -----------------------------
dat <- read.csv(data_path, check.names = TRUE)
sig <- read.csv(sig_path, check.names = TRUE)

## Expect columns "y" and "mediator" in sig
stopifnot(all(c("y", "mediator") %in% names(sig)))

## Make sure exposure exists
stopifnot("gene" %in% names(dat))

## Treat gene as numeric (0/1/2 etc.)
if (is.factor(dat$gene)) {
  dat$gene <- as.numeric(as.character(dat$gene))
}
stopifnot(is.numeric(dat$gene))

## -----------------------------
## 3. Define variables
## -----------------------------
treat_var <- "gene"

## Covariates – keep consistent with your main grid
covars_raw <- c(
  "age", "sex", "education",
  "learning_deficits",
  "HSV_r", "BMI",
  "hypertension", "thyroid_diseases",
  "other_diseases", "dementia_history_parents"
)

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
## 4. Helper: choose family for Y
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
## 5. Loop over significant pairs
## -----------------------------
med_fits <- list()

for (i in seq_len(nrow(sig))) {
  y_name <- as.character(sig$y[i])
  m_name <- as.character(sig$mediator[i])

  cat("\n========================================\n")
  cat("Pair", i, "of", nrow(sig), "\n")
  cat("Outcome :", y_name, "\n")
  cat("Mediator:", m_name, "\n")
  cat("========================================\n")

  ## Basic existence checks
  stopifnot(y_name %in% names(dat))
  stopifnot(m_name %in% names(dat))

  fam_info <- choose_family_y(dat[[y_name]])
  fam_y    <- fam_info$fam

  ## Build subset with needed variables
  vars_needed <- c(treat_var, y_name, m_name, covars)
  vars_needed <- vars_needed[vars_needed %in% names(dat)]
  stopifnot(length(vars_needed) >= 3)  # at least gene, y, m

  df_sub <- dat[complete.cases(dat[, vars_needed, drop = FALSE]), vars_needed, drop = FALSE]

  ## Sanity checks (fail fast)
  if (nrow(df_sub) < 40) {
    stop(paste0("Not enough complete rows (n = ", nrow(df_sub), ") for y=", y_name,
                ", mediator=", m_name))
  }
  if (var(df_sub[[treat_var]], na.rm = TRUE) == 0) {
    stop(paste("No variation in 'gene' for y =", y_name, "mediator =", m_name))
  }
  if (var(df_sub[[m_name]], na.rm = TRUE) == 0) {
    stop(paste("Mediator has zero variance for y =", y_name, "mediator =", m_name))
  }

  ## Formulas
  rhs_out <- c(treat_var, m_name, covars)
  f_out <- as.formula(paste(y_name, "~", paste(rhs_out, collapse = " + ")))

  rhs_med <- c(treat_var, covars)
  f_med <- as.formula(paste(m_name, "~", paste(rhs_med, collapse = " + ")))

  ## Fit models
  out_model <- glm(f_out, data = df_sub, family = fam_y)
  med_model <- glm(f_med, data = df_sub, family = gaussian())

  ## Define treatment / control values for gene
  tmin <- min(df_sub[[treat_var]], na.rm = TRUE)
  tmax <- max(df_sub[[treat_var]], na.rm = TRUE)
  if (isTRUE(all.equal(tmin, tmax))) {
    stop(paste("gene is constant in subset for y =", y_name, "mediator =", m_name))
  }

  ## Run mediation (more sims for more stable CIs, change if needed)
  med_fit <- mediate(
    model.m       = med_model,
    model.y       = out_model,
    treat         = treat_var,
    mediator      = m_name,
    treat.value   = tmax,
    control.value = tmin,
    sims          = 10000
  )

  ## Print a concise summary for this pair
  print(summary(med_fit))

  ## Store object in list for later use
  key <- paste(y_name, m_name, sep = "__")
  med_fits[[key]] <- med_fit
}

cat("\nFinished all significant pairs.\n")
cat("Stored", length(med_fits), "mediate objects in 'med_fits'.\n")
