library(tidyverse)
df <- read.csv("got_scripts_breakdown.csv", sep=";", header = T)
df <- as_tibble(df)
