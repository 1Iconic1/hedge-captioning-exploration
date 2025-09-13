################################################################################
### Setup ###
################################################################################
########################## LIBRARIES #############################
library(tidyverse)
library(scales)
library(broom)
library(car)
library(effects)
options(scipen = 999)
options(digits = 9)
options(pillar.sigfig = 6)

# for output formatting
library(dplyr)
library(purrr)
library(knitr)
library(kableExtra)
library(stringr)

########################## HELPER FUNCTIONS #############################
# --- Function for stars ---
get_stars <- function(p) {
  if (is.na(p)) {
    return("")
  }
  if (p < 0.001) {
    "***"
  } else if (p < 0.01) {
    "**"
  } else if (p < 0.05) {
    "*"
  } else {
    ""
  }
}

# --- Format + align numbers ---
format_estimate <- function(estimate, p.value, digits = 4) {
  stars <- get_stars(p.value)
  num_str <- sprintf(paste0("%.", digits, "f"), estimate)
  paste0(num_str, stars)
}

align_column <- function(x) {
  max_len <- max(nchar(x), na.rm = TRUE)
  str_pad(x, width = max_len, side = "right", pad = " ")
}

# --- Pretty coefficient labels ---
pretty_term <- function(term) {
  term <- str_replace_all(term, "`", "") # strip backticks

  # Replace pieces inside interaction terms too
  term <- str_replace_all(term, "blur1", "Blur = True")
  term <- str_replace_all(term, "framing1", "Framing = True")
  term <- str_replace_all(term, "rotation1", "Rotation = True")
  term <- str_replace_all(term, "curved_label1", "Rounded Label = True")
  term <- str_replace_all(term, "text_panel1", "Text Panel = True")
  term <- str_replace_all(term, ":", " AND ")
  term <- str_replace_all(term, "model", "Model = ")
  term <- str_replace_all(term, "gemini-2.5-flash", "Gemini")
  term <- str_replace_all(term, "llama-90b-4bit", "Llama")
  term <- str_replace_all(term, "molmo-72b-4bit", "Molmo")
  term
}

# --- Universal Table Function ---
regression_table <- function(models, model_names = NULL, digits_coef = 4, digits_dev = 1) {
  if (!is.list(models)) models <- list(models)
  if (is.null(model_names)) model_names <- paste0("Model ", seq_along(models))

  model_tabs <- map2(models, model_names, function(model, mname) {
    coef_df <- broom::tidy(model) %>%
      mutate(
        value = map2_chr(estimate, p.value, ~ format_estimate(.x, .y, digits_coef)),
        term = map_chr(term, pretty_term)
      ) %>%
      # mutate(value = align_column(value)) %>%
      select(term, !!mname := value)

    fit_rows <- tibble(
      term = c(
        sprintf("Null deviance (df = %d)", model$df.null),
        sprintf("Residual deviance (df = %d)", model$df.residual),
        "AIC"
      ),
      !!mname := c(
        sprintf(paste0("%.", digits_dev, "f"), model$null.deviance),
        sprintf(paste0("%.", digits_dev, "f"), model$deviance),
        sprintf(paste0("%.", digits_dev, "f"), AIC(model))
      )
    )

    bind_rows(coef_df, fit_rows)
  })

  # Merge on term
  final_tab <- reduce(model_tabs, full_join, by = "term") %>%
    rename(`Independent Variable` = term)
  final_tab
}

# Function to build nice table for one model
one_model_table <- function(model, digits_coef = 4, digits_dev = 1) {
  coef_df <- broom::tidy(model) %>%
    mutate(
      stars = map_chr(p.value, get_stars),
      value = paste0(sprintf(paste0("%.", digits_coef, "f"), estimate), stars),
      term = map_chr(term, pretty_term)
    ) %>%
    select(term, value)

  # Fit statistics
  fit_rows <- tibble(
    term = c(
      sprintf("Null deviance (df = %d)", model$df.null),
      sprintf("Residual deviance (df = %d)", model$df.residual),
      "AIC"
    ),
    value = c(
      sprintf(paste0("%.", digits_dev, "f"), model$null.deviance),
      sprintf(paste0("%.", digits_dev, "f"), model$deviance),
      sprintf(paste0("%.", digits_dev, "f"), AIC(model))
    )
  )

  final_tab <- bind_rows(coef_df, fit_rows) %>%
    rename(
      `Independent Variable` = term,
      Estimate = value
    )
  final_tab

  # Pretty print
  # kable(final_tab, align = c("l", "c"), booktabs = TRUE, escape = FALSE) %>%
  #   kable_styling(latex_options = c("hold_position"))
}

compute_odds_from_estimate <- function(estimate) {
  # exp(as.numeric(str_replace_all(estimate, "[*.]+$", "")))
  round(100 * plogis(as.numeric(str_replace_all(estimate, "[*.]+$", ""))), 2)
}
########################## SINGLE REGRESSION MODEL #############################
# load data from csv
annotated_regression_two_bins_file <- "./regression-data/regression-final-df_1859-images.csv"
annotated_regression_two_bins_df <- readr::read_csv(annotated_regression_two_bins_file)
annotated_regression_two_bins_df

# isolate the variables we want
filtered_regression_df <- annotated_regression_two_bins_df[!names(annotated_regression_two_bins_df)
%in% c(
    "id", "file_name", "image_url",
    "text_detected", "unrecognizable", "other", "no issue",
    "human_captions", "gpt4o_caption", "llama_caption", "molmo_caption"
  )]

factor_cols <- c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
iv_factor_cols <- c("type", "curved label", "text panel", "framing", "blur", "rotation")
dv_factor_cols <- c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
filtered_regression_df[iv_factor_cols] <- lapply(filtered_regression_df[iv_factor_cols], function(x) factor(x))
filtered_regression_df[dv_factor_cols] <- lapply(filtered_regression_df[factor_cols], factor)

long_df <- filtered_regression_df %>%
  pivot_longer(
    cols = c(`gpt-4.1_correct`, `gemini-2.5-flash_correct`, `llama-90b-4bit_correct`, `molmo-72b-4bit_correct`),
    names_to = c("model", ".value"),
    names_sep = "_"
  )
long_df$correct <- as.factor(long_df$correct)
long_df$model <- as.factor(as.character(long_df$model))
long_df$model <- factor(long_df$model, levels = c("gpt-4.1", "gemini-2.5-flash", "llama-90b-4bit", "molmo-72b-4bit"))
long_df

# rename curved panel and text panel
long_df <- long_df %>%
  dplyr::rename(
    curved_label = `curved label`,
    text_panel   = `text panel`
  )

## interaction model
model_interaction <- glm(
  correct ~
    blur * framing * rotation +
    blur * curved_label +
    blur * text_panel +
    framing * curved_label +
    framing * text_panel +
    rotation * curved_label +
    rotation * text_panel +
    framing * model +
    blur * model +
    rotation * model,
  data = long_df,
  family = binomial(link = "logit")
)
summary(model_interaction)
confint(model_interaction)
car::vif(model_interaction, type = "predictor")

# add odds ratio
single_reg_output <- one_model_table(model_interaction)
single_reg_output <- single_reg_output %>%
  rowwise() %>%
  mutate(corrected_pct = compute_odds_from_estimate(Estimate)) %>%
  mutate(pct_change = round(100 * (1 - odds_ratio), digits = 2))
print(single_reg_output, n = 100)

# Pretty print
kable(single_reg_output, align = c("l", "c"), booktabs = TRUE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position"))

# formatted table
single_reg_output <- one_model_table(model_interaction)
kable(single_reg_output, align = c("l", "c"), booktabs = TRUE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position"))

# effect plots for significant variables
plot(effect("blur:framing", model_interaction),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:rotation", model_interaction),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("framing:rotation", model_interaction),
  multiline = TRUE, ci.style = "bands"
)

plot(effect("framing:text_panel", model_interaction),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("rotation:curved_label", model_interaction),
  multiline = TRUE, ci.style = "bands"
)

plot(effect("blur:model", model_interaction),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("rotation:model", model_interaction),
  multiline = TRUE, ci.style = "bands"
)
########################### REGRESSION PER MODEL ###############################
# load data from csv
annotated_regression_two_bins_file <- "./regression-data/regression-final-df_1859-images.csv"
annotated_regression_two_bins_df <- readr::read_csv(annotated_regression_two_bins_file)
annotated_regression_two_bins_df

# isolate the variables we want
filtered_regression_df <- annotated_regression_two_bins_df[!names(annotated_regression_two_bins_df)
%in% c(
    "id", "file_name", "image_url",
    "text_detected", "unrecognizable", "other", "no issue",
    "human_captions", "gpt4o_caption", "llama_caption", "molmo_caption"
  )]

factor_cols <- c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
iv_factor_cols <- c("type", "curved label", "text panel", "framing", "blur", "rotation")
filtered_regression_df[factor_cols] <- lapply(filtered_regression_df[factor_cols], factor)
filtered_regression_df[iv_factor_cols] <- lapply(filtered_regression_df[iv_factor_cols], function(x) factor(x))

# rename curved panel and text panel and correct columns
filtered_regression_df <- filtered_regression_df %>%
  dplyr::rename(
    curved_label = `curved label`,
    text_panel = `text panel`,
    gpt.correct = `gpt-4.1_correct`,
    gemini.correct = `gemini-2.5-flash_correct`,
    llama.correct = `llama-90b-4bit_correct`,
    molmo.correct = `molmo-72b-4bit_correct`
  )
filtered_regression_df

# separate into 4 dataframes
gpt_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gemini.correct", "llama.correct", "molmo.correct")
]
gemini_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt.correct", "llama.correct", "molmo.correct")
]
llama_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt.correct", "gemini.correct", "molmo.correct")
]
molmo_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt.correct", "gemini.correct", "llama.correct")
]

# Named list of your datasets
datasets <- list(
  gpt = gpt_regression_df,
  gemini = gemini_regression_df,
  llama = llama_regression_df,
  molmo = molmo_regression_df
)

# Named list of outcome variables
outcomes <- c(
  gpt = "gpt.correct",
  gemini = "gemini.correct",
  llama = "llama.correct",
  molmo = "molmo.correct"
)

# model formula
formula <- "blur * framing * rotation"

# Run GLMs in a loop
results <- list()
for (dataset_name in names(datasets)) {
  formula_full <- as.formula(paste(outcomes[dataset_name], "~", formula))
  glm_result <- glm(
    formula_full,
    data = datasets[[dataset_name]],
    family = binomial(link = "logit")
  )
  # Store results with descriptive names
  results[[dataset_name]] <- glm_result
}

# nicely formatted table
models <- list(
  results$gpt,
  results$gemini,
  results$llama,
  results$molmo
)
all_models <- regression_table(
  models,
  model_names = c("GPT", "Gemini", "Llama", "Molmo")
)
all_models

# add odds ratio
all_models_odds <- all_models %>%
  rowwise() %>%
  mutate(`GPT Odds Ratio` = compute_odds_from_estimate(GPT)) %>%
  mutate(`Gemini Odds Ratio` = compute_odds_from_estimate(Gemini)) %>%
  mutate(`Llama Odds Ratio` = compute_odds_from_estimate(Llama)) %>%
  mutate(`Molmo Odds Ratio` = compute_odds_from_estimate(Molmo)) %>%
  mutate(`GPT Pct Change` = round(100 * (1 - `GPT Odds Ratio`), digits = 2)) %>%
  mutate(`Gemini Pct Change` = round(100 * (1 - `Gemini Odds Ratio`), digits = 2)) %>%
  mutate(`Llama Pct Change` = round(100 * (1 - `Llama Odds Ratio`), digits = 2)) %>%
  mutate(`Molmo Pct Change` = round(100 * (1 - `Molmo Odds Ratio`), digits = 2))

# Kable output with monospace so padding works
kable(all_models_odds,
  align = c("l", rep("c", length(models))),
  booktabs = TRUE, escape = FALSE
) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  column_spec(2:(length(models) + 1), monospace = TRUE)
# write_csv(all_modes, "./regression-data/output-individual-models.csv")

summary(results$gpt)
summary(results$gemini)
summary(results$llama)
summary(results$molmo)

############################### SEPARATE INDIVIDUAL MODELS ##################################
# these are the same as above, but done individually so assumptions / effects can be analyzed
# load data from csv
annotated_regression_two_bins_file <- "./regression-data/regression-final-df_1859-images.csv"
annotated_regression_two_bins_df <- readr::read_csv(annotated_regression_two_bins_file)
annotated_regression_two_bins_df

# isolate the variables we want
filtered_regression_df <- annotated_regression_two_bins_df[!names(annotated_regression_two_bins_df)
%in% c(
    "id", "file_name", "image_url",
    "text_detected", "unrecognizable", "other", "no issue",
    "human_captions", "gpt4o_caption", "llama_caption", "molmo_caption"
  )]

factor_cols <- c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
iv_factor_cols <- c("type", "curved label", "text panel", "framing", "blur", "rotation")
filtered_regression_df[factor_cols] <- lapply(filtered_regression_df[factor_cols], factor)
filtered_regression_df[iv_factor_cols] <- lapply(filtered_regression_df[iv_factor_cols], function(x) factor(x))

# rename curved panel and text panel and correct columns
filtered_regression_df <- filtered_regression_df %>%
  dplyr::rename(
    curved_label = `curved label`,
    text_panel = `text panel`,
    gpt.correct = `gpt-4.1_correct`,
    gemini.correct = `gemini-2.5-flash_correct`,
    llama.correct = `llama-90b-4bit_correct`,
    molmo.correct = `molmo-72b-4bit_correct`
  )
filtered_regression_df

# separate into 4 dataframes
gpt_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gemini.correct", "llama.correct", "molmo.correct")
]
gemini_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt.correct", "llama.correct", "molmo.correct")
]
llama_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt.correct", "gemini.correct", "molmo.correct")
]
molmo_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt.correct", "gemini.correct", "llama.correct")
]

## gpt model with interactions
gpt_model_interactions <- glm(
  gpt.correct ~
    blur * framing * rotation,
  data = gpt_regression_df,
  family = binomial(link = "logit")
)
summary(gpt_model_interactions)
confint(gpt_model_interactions)

plot(effect("blur:framing", gpt_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:rotation", gpt_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("framing:rotation", gpt_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:framing:rotation", gpt_model_interactions),
  multiline = TRUE, ci.style = "bands"
)

## gemini model with interactions
gemini_model_interactions <- glm(
  gemini.correct ~
    blur * framing * rotation,
  data = gemini_regression_df,
  family = binomial(link = "logit")
)
summary(gemini_model_interactions)
confint(gemini_model_interactions)

plot(effect("blur:framing", gemini_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:rotation", gemini_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("framing:rotation", gemini_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:framing:rotation", gemini_model_interactions),
  multiline = TRUE, ci.style = "bands"
)

## llama model with interactions
llama_model_interactions <- glm(
  llama.correct ~
    blur * framing * rotation,
  data = llama_regression_df,
  family = binomial(link = "logit")
)
summary(llama_model_interactions)
confint(llama_model_interactions)

plot(effect("blur:framing", llama_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:rotation", llama_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("framing:rotation", llama_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:framing:rotation", llama_model_interactions),
  multiline = TRUE, ci.style = "bands"
)

## molmo model with interactions
molmo_model_interactions <- glm(
  molmo.correct ~
    blur * framing * rotation,
  data = molmo_regression_df,
  family = binomial(link = "logit")
)
summary(molmo_model_interactions)
confint(molmo_model_interactions)

plot(effect("blur:framing", molmo_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:rotation", molmo_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("framing:rotation", molmo_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
plot(effect("blur:framing:rotation", molmo_model_interactions),
  multiline = TRUE, ci.style = "bands"
)
