################################################################################
### Setup ###
################################################################################
# load libraries
library(tidyverse)
library(scales)
options(scipen = 999)

# load data from csv
high_quality_data_file <- "./intermediate-data/study1_high-quality-data-for-statistical-testing.csv"
low_quality_data_file <- "./intermediate-data/study1_low-quality-data-for-statistical-testing.csv"

high_quality_df <- readr::read_csv(high_quality_data_file)
low_quality_df <- readr::read_csv(low_quality_data_file)

colnames(high_quality_df)


# conduct Mann Whitney U test for each model
high_quality_df["gpt-4o-2024-08-06_pbert"]
high_quality_df["gpt-4o-2024-08-06_rbert"]
high_quality_df["gpt-4o-2024-08-06_fbert"]

high_low_gpt <- data.frame(
  "high_precision" = c(high_quality_df["gpt-4o-2024-08-06_pbert"], rep(NA, nrow(low_quality_df) - nrow(high_quality_df))),
  "high_recall" = c(high_quality_df["gpt-4o-2024-08-06_rbert"], rep(NA, nrow(low_quality_df) - nrow(high_quality_df))),
  "high_f1" = c(high_quality_df["gpt-4o-2024-08-06_fbert"], rep(NA, nrow(low_quality_df) - nrow(high_quality_df))),
  "low_precision" = low_quality_df["gpt-4o-2024-08-06_pbert"],
  "low_recall" = low_quality_df["gpt-4o-2024-08-06_rbert"],
  "low_f1" = low_quality_df["gpt-4o-2024-08-06_fbert"]
)
high_low_gpt

c(high_quality_df["gpt-4o-2024-08-06_pbert"], rep(NA, nrow(low_quality_df) - nrow(high_quality_df)))

high_quality_df["gpt-4o-2024-08-06_pbert"]

c(as.vector(high_quality_df["gpt-4o-2024-08-06_pbert"]), rep(NA, nrow(low_quality_df) - nrow(high_quality_df)))

high_quality_df["gpt-4o-2024-08-06_pbert"]

cbind.fill <- function(..., names = NA) {
  xlist <- list(...)
  y <- Reduce(
    function(a, b) {
      if (is.vector(a)) na <- length(a)
      if (is.data.frame(a) | is.matrix(a)) na <- nrow(a)
      if (is.vector(b)) nb <- length(b)
      if (is.data.frame(b) | is.matrix(b)) nb <- nrow(b)
      subset(
        merge(
          cbind(cbindfill.id = 1:na, a),
          cbind(cbindfill.id = 1:nb, b),
          all = TRUE, by = "cbindfill.id"
        ),
        select = -cbindfill.id
      )
    },
    xlist
  )
  if (!is.na(names[1])) colnames(y) <- names
  return(y)
}
precision <- cbind.fill(high_quality_df["gpt-4o-2024-08-06_pbert"], low_quality_df["gpt-4o-2024-08-06_pbert"])
precision$high <- precision$`gpt-4o-2024-08-06_pbert.x`
precision$low <- precision$`gpt-4o-2024-08-06_pbert.y`
wilcox.test(high ~ low, data = precision)

t.test(high ~ low, data = precision)
precision


# linear regression
colnames(low_quality_df)

### gpt
precision.gpt.fiti <- lm(`gpt-4o-2024-08-06_pbert` ~
  framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
  blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
  obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
  rotation * `too dark` + rotation * `too bright` +
  `too bright` * `too dark`, data = low_quality_df)
summary(precision.gpt.fiti)

recall.gpt.fiti <- lm(`gpt-4o-2024-08-06_rbert` ~
  framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
  blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
  obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
  rotation * `too dark` + rotation * `too bright` +
  `too bright` * `too dark`, data = low_quality_df)
summary(recall.gpt.fiti)

### llama
precision.llama.fiti <- lm(`Llama-3.2-11B-Vision-Instruct_pbert` ~
  framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
  blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
  obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
  rotation * `too dark` + rotation * `too bright` +
  `too bright` * `too dark`, data = low_quality_df)
summary(precision.llama.fiti)

recall.llama.fiti <- lm(`Llama-3.2-11B-Vision-Instruct_rbert` ~
  framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
  blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
  obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
  rotation * `too dark` + rotation * `too bright` +
  `too bright` * `too dark`, data = low_quality_df)
summary(recall.llama.fiti)

### Molmo
precision.molmo.fiti <- lm(`Molmo-7B-O-0924_pbert` ~
  framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
  blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
  obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
  rotation * `too dark` + rotation * `too bright` +
  `too bright` * `too dark`, data = low_quality_df)
summary(precision.molmo.fiti)

recall.molmo.fiti <- lm(`Molmo-7B-O-0924_rbert` ~
  framing * blur + framing * obstruction + framing * rotation + framing * `too dark` + framing * `too bright` +
  blur * obstruction + blur * rotation + blur * `too dark` + blur * `too bright` +
  obstruction * rotation + obstruction * `too dark` + obstruction * `too bright` +
  rotation * `too dark` + rotation * `too bright` +
  `too bright` * `too dark`, data = low_quality_df)
summary(recall.molmo.fiti)
