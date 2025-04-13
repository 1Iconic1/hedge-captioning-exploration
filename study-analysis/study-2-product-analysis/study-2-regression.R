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
annotated_regression_file <- "./study-2-product-analysis/intermediate-data/regression-df_04-13-25-00:00.csv"
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
gpt_regression_df <- filtered_regression_df[
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

################################################################################
## simple gpt model without interactions
gpt_model_simple <- glm(
  gpt4o_code ~ framing + blur + obstruction + rotation + `too dark` + `too bright`,
  data = gpt_regression_df,
  family = binomial
)
summary(gpt_model_simple)
confint(gpt_model_simple)

### analysis of assumptions
probabilities <- predict(gpt_model_simple, type = "response")
gpt_regression_df_linearity <- gpt_regression_df %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(gpt_regression_df_linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")
plot(gpt_model_simple, which = 4, id.n = 3)
gpt_model_simple.data <- augment(gpt_model_simple) %>% 
  mutate(index = 1:n()) 
gpt_model_simple.data %>% filter(abs(.std.resid) > 3)
car::vif(gpt_model_simple, type = 'predictor')

## gpt model with interactions
gpt_model_interactions <- glm(
  gpt4o_code ~
    framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
    blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
    obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
    rotation * `too dark` + rotation * `too bright` +
    `too bright` * `too dark`,
  data = gpt_regression_df,
  family = binomial
)
summary(gpt_model_interactions)
confint(gpt_model_interactions)

### analysis of assumptions
probabilities <- predict(gpt_model_interactions, type = "response")
gpt_regression_df_linearity <- gpt_regression_df %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(gpt_regression_df_linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")
plot(gpt_model_interactions, which = 4, id.n = 3)
gpt_model_interactions.data <- augment(gpt_model_interactions) %>% 
  mutate(index = 1:n()) 
gpt_model_interactions.data %>% filter(abs(.std.resid) > 3)
car::vif(gpt_model_interactions, type = 'predictor')

################################################################################
## simple llama model without interactions
llama_model_simple <- glm(
  llama_code ~ framing + blur + obstruction + rotation + `too dark` + `too bright`,
  data = llama_regression_df,
  family = binomial
)
summary(llama_model_simple)
confint(llama_model_simple)

### analysis of assumptions
probabilities <- predict(llama_model_simple, type = "response")
llama_regression_df_linearity <- llama_regression_df %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(llama_regression_df_linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")
plot(llama_model_simple, which = 4, id.n = 3)
llama_model_simple.data <- augment(llama_model_simple) %>% 
  mutate(index = 1:n()) 
llama_model_simple.data %>% filter(abs(.std.resid) > 3)
car::vif(llama_model_simple, type = 'predictor')

## llama model with interactions
llama_model_interactions <- glm(
  llama_code ~
    framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
    blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
    obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
    rotation * `too dark` + rotation * `too bright` +
    `too bright` * `too dark`,
  data = llama_regression_df,
  family = binomial
)
summary(llama_model_interactions)
confint(llama_model_interactions)

### analysis of assumptions
probabilities <- predict(llama_model_interactions, type = "response")
llama_regression_df_linearity <- llama_regression_df %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(llama_regression_df_linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")
plot(llama_model_interactions, which = 4, id.n = 3)
llama_model_interactions.data <- augment(llama_model_interactions) %>% 
  mutate(index = 1:n()) 
llama_model_interactions.data %>% filter(abs(.std.resid) > 3)
car::vif(llama_model_interactions, type = 'predictor')

################################################################################
## simple molmo model without interactions
molmo_model_simple <- glm(
  molmo_code ~ framing + blur + obstruction + rotation + `too dark` + `too bright`,
  data = molmo_regression_df,
  family = binomial
)
summary(molmo_model_simple)
confint(molmo_model_simple)

### analysis of assumptions
probabilities <- predict(molmo_model_simple, type = "response")
molmo_regression_df_linearity <- molmo_regression_df %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(molmo_regression_df_linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")
plot(molmo_model_simple, which = 4, id.n = 3)
molmo_model_simple.data <- augment(molmo_model_simple) %>% 
  mutate(index = 1:n()) 
molmo_model_simple.data %>% filter(abs(.std.resid) > 3)
car::vif(molmo_model_simple, type = 'predictor')

## molmo model with interactions
molmo_model_interactions <- glm(
  molmo_code ~
    framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
    blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
    obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
    rotation * `too dark` + rotation * `too bright` +
    `too bright` * `too dark`,
  data = molmo_regression_df,
  family = binomial
)
summary(molmo_model_interactions)
confint(molmo_model_interactions)

### analysis of assumptions
probabilities <- predict(molmo_model_interactions, type = "response")
molmo_regression_df_linearity <- molmo_regression_df %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(molmo_regression_df_linearity, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")
plot(molmo_model_interactions, which = 4, id.n = 3)
molmo_model_interactions.data <- augment(molmo_model_interactions) %>% 
  mutate(index = 1:n()) 
molmo_model_interactions.data %>% filter(abs(.std.resid) > 3)
car::vif(molmo_model_interactions, type = 'predictor')
################################################################################