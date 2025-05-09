################################################################################
### Setup ###
################################################################################
# load libraries
library(tidyverse)
library(scales)
library(broom)
library(car)
options(scipen = 999)

# load data from csv
annotated_regression_file <- "../experiments/sampled-data-consistency_200-examples_5-samples.csv"
annotated_regression_df <- readr::read_csv(annotated_regression_file)
annotated_regression_df

# isolate the variables we want
filtered_regression_df <- annotated_regression_df[!names(annotated_regression_df)
                                                  %in% c(
                                                    "image_id", "file_name", "vizwiz_url",
                                                    "text_detected", "unrecognizable", "other", "no issue",
                                                    "human_captions", "gpt4o_caption", "llama_caption", "molmo_caption"
                                                  )]

factor_cols <- c("gpt4o_code", "llama_code", "molmo_code")
filtered_regression_df[factor_cols] <- lapply(filtered_regression_df[factor_cols], factor)

# separate into 3 dataframes: gpt, llama, molmo
gpt4o_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("llama_code", "molmo_code")
]
llama_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt4o_code", "molmo_code")
]
molmo_regression_df <- filtered_regression_df[
  !names(filtered_regression_df) %in%
    c("gpt4o_code", "llama_code")
]

# create model
predictors <- c(
  "framing", "blur", "obstruction",
  "rotation", "too dark", "too bright"
)

################################## CONTINUOUS ##################################
## simple gpt4o model without interactions
gpt4o_model_simple <- lm(
  gpt4o_nli_avg_contrad ~ framing + blur + obstruction + rotation + `too dark` + `too bright` + gpt4o_code,
  data = gpt4o_regression_df
)
summary(gpt4o_model_simple)
confint(gpt4o_model_simple)

### analysis of assumptions
gpt4o_model_simple.diag.metrics <- augment(gpt4o_model_simple)
par(mfrow = c(2, 2))
plot(gpt4o_model_simple)
car::vif(gpt4o_model_simple)

## simple llama model without interactions
llama_model_simple <- lm(
  llama_nli_avg_contrad ~ framing + blur + obstruction + rotation + `too dark` + `too bright` + llama_code,
  data = llama_regression_df
)
summary(llama_model_simple)
confint(llama_model_simple)

### analysis of assumptions
### residuals don't look normal
llama_model_simple.diag.metrics <- augment(llama_model_simple)
par(mfrow = c(2, 2))
plot(llama_model_simple)
car::vif(llama_model_simple)

## simple molmo model without interactions
molmo_model_simple <- lm(
  molmo_nli_avg_contrad ~ framing + blur + obstruction + rotation + `too dark` + `too bright` + molmo_code,
  data = molmo_regression_df
)
summary(molmo_model_simple)
confint(molmo_model_simple)

### analysis of assumptions
### residuals also not normal in upper quartiles
molmo_model_simple.diag.metrics <- augment(molmo_model_simple)
par(mfrow = c(2, 2))
plot(molmo_model_simple)
car::vif(molmo_model_simple)
