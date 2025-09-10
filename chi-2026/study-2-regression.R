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
  term <- str_replace_all(term, "curved label1", "Rounded Label = True")
  term <- str_replace_all(term, "text panel1", "Text Panel = True")
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

  # Kable output with monospace so padding works
  kable(final_tab,
    align = c("l", rep("c", length(models))),
    booktabs = TRUE, escape = FALSE
  ) %>%
    kable_styling(latex_options = c("hold_position")) %>%
    column_spec(2:(length(models) + 1), monospace = TRUE)
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

## simple model
model_simple <- glm(
  correct ~ framing * blur * rotation * model + framing * blur * rotation * `curved label` * `text panel`,
  data = long_df,
  family = binomial(link = "logit")
)
summary(model_simple)
confint(model_simple)
car::vif(model_simple, type = "predictor")

## interaction model
model_interaction <- glm(
  correct ~
    blur * framing * rotation +
    blur * `curved label` +
    blur * `text panel` +
    framing * `curved label` +
    framing * `text panel` +
    rotation * `curved label` +
    rotation * `text panel` +
    framing * model +
    blur * model +
    rotation * model,
  data = long_df,
  family = binomial(link = "logit")
)
summary(model_interaction)
confint(model_interaction)
car::vif(model_interaction, type = "predictor")

# formatted table
one_model_table(model_interaction)

########################### REGRESSION PER MODEL ###############################
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

  # Pretty print
  kable(final_tab, align = c("l", "c"), booktabs = TRUE, escape = FALSE) %>%
    kable_styling(latex_options = c("hold_position"))
}

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

# get only low-quality data
# filtered_regression_df = filtered_regression_df[filtered_regression_df$type == 'low-quality', ]

factor_cols <- c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
iv_factor_cols <- c("type", "curved label", "text panel", "framing", "blur", "rotation")
filtered_regression_df[factor_cols] <- lapply(filtered_regression_df[factor_cols], factor)
filtered_regression_df[iv_factor_cols] <- lapply(filtered_regression_df[iv_factor_cols], function(x) factor(x))

# separate into 4 dataframes
gpt_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
]
gemini_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt-4.1_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
]
llama_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt-4.1_correct", "gemini-2.5-flash_correct", "molmo-72b-4bit_correct")
]
molmo_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct")
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
  gpt = "`gpt-4.1_correct`",
  gemini = "`gemini-2.5-flash_correct`",
  llama = "`llama-90b-4bit_correct`",
  molmo = "`molmo-72b-4bit_correct`"
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
all_modes <- regression_table(
  models = list(
    results$gpt,
    results$gemini,
    results$llama,
    results$molmo
  ),
  model_names = c("GPT", "Gemini", "Llama", "Molmo")
)
all_modes
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

# separate into 4 dataframes
gpt_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gemini-2.5-flash_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
]
gemini_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt-4.1_correct", "llama-90b-4bit_correct", "molmo-72b-4bit_correct")
]
llama_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt-4.1_correct", "gemini-2.5-flash_correct", "molmo-72b-4bit_correct")
]
molmo_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt-4.1_correct", "gemini-2.5-flash_correct", "llama-90b-4bit_correct")
]

## gpt model with interactions
gpt_model_interactions <- glm(
  `gpt-4.1_correct` ~
    blur * framing * rotation,
  data = gpt_regression_df,
  family = binomial(link = "logit")
)
summary(gpt_model_interactions)
confint(gpt_model_interactions)

## gemini model with interactions
gemini_model_interactions <- glm(
  `gemini-2.5-flash_correct` ~
    blur * framing * rotation,
  data = gemini_regression_df,
  family = binomial(link = "logit")
)
summary(gemini_model_interactions)
confint(gemini_model_interactions)

## llama model with interactions
llama_model_interactions <- glm(
  `llama-90b-4bit_correct` ~
    blur * framing * rotation,
  data = llama_regression_df,
  family = binomial(link = "logit")
)
summary(llama_model_interactions)
confint(llama_model_interactions)

## molmo model with interactions
molmo_model_interactions <- glm(
  `molmo-72b-4bit_correct` ~
    blur * framing * rotation,
  data = molmo_regression_df,
  family = binomial(link = "logit")
)
summary(molmo_model_interactions)
confint(molmo_model_interactions)
############################### OLD CODE ##################################
#
# # multi-model version
# extract_model_info <- function(model, model_name) {
#   coef_df <- broom::tidy(model) %>%
#     mutate(stars = map_chr(p.value, get_stars),
#            estimate_fmt = paste0(sprintf("%.4f", estimate), stars),
#            term = map_chr(term, pretty_term)) %>%
#     select(term, estimate_fmt)
#
#   stats_df <- tibble(
#     term = c(
#       sprintf("Null deviance (df = %d)", model$df.null),
#       sprintf("Residual deviance (df = %d)", model$df.residual),
#       "AIC"
#     ),
#     estimate_fmt = c(
#       sprintf("%.1f", model$null.deviance),
#       sprintf("%.1f", model$deviance),
#       sprintf("%.1f", AIC(model))
#     )
#   )
#
#   bind_rows(coef_df, stats_df) %>%
#     rename(!!model_name := estimate_fmt)
# }
#
# multi_model_table <- function(models) {
#   model_tables <- imap(models, extract_model_info)
#   final_tab <- reduce(model_tables, full_join, by = "term") %>%
#     rename(`Independent Variable` = term)
#
#   kable(final_tab, align = c("l", rep("c", length(models))),
#         booktabs = TRUE, escape = FALSE) %>%
#     kable_styling(latex_options = c("hold_position"))
# }


############################### LONG REG ##################################
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

## simple model
model_simple <- glm(
  correct ~ framing * blur * rotation * model + framing * blur * rotation * `curved label` * `text panel`,
  data = long_df,
  family = binomial(link = "logit")
)
summary(model_simple)
confint(model_simple)
car::vif(model_simple, type = "predictor")

## interaction model
model_interaction <- glm(
  correct ~
    blur * framing * rotation +
    blur * `curved label` +
    blur * `text panel` +
    framing * `curved label` +
    framing * `text panel` +
    rotation * `curved label` +
    rotation * `text panel` +
    framing * model +
    blur * model +
    rotation * model,
  data = long_df,
  family = binomial(link = "logit")
)
summary(model_interaction)
confint(model_interaction)
car::vif(model_interaction, type = "predictor")

# formatted table
one_model_table(model_interaction)
