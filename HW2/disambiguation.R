# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# disambiguation.py: Welcome to my R script. I assume you came my python
#   script so we'll skip the chit chat. Your standard CS gods have no power
#   here, this is no man's land.
#
# Author: April Walker
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
library(h2o)

# indicate wordpairs of interest here
word_pairs <- list(c("night", "seat"),
                   c("kitchen", "cough"),
                   c("car", "bike"),
                   c("manufacturer", "bike"),
                   c("big", "small"),
                   c("huge", "heavy"))

# # #
# Step 1: Dataset setup - this loop is very similar to the one ~ line 125 in R
#   however all train/test data is instead saved in a big ole list
# # #
train <- test <- list() # no truth set used for this code

for(pair in word_pairs) {
  # declare train/test sets for this wordpair
  pair_name <- paste(pair, collapse="")
  train[[pair_name]] <- test[[pair_name]] <- list()

  for(word in pair) {
    # declare filenames to grab context
    train_file <- paste(word, "training.txt", sep="_")
    test_file <- paste(word, "testing.txt", sep="_")
    # get context
    new_train <- read.table(train_file, header=F)
    new_test <- read.table(test_file, header=F)
    # append true target value
    new_train["target"] = word
    new_test["target"] = word
    # append dataframes to train/truth sets
    train[[pair_name]] <- rbind(train[[pair_name]], new_train)
    test[[pair_name]] <- rbind(test[[pair_name]], new_test)
  }
  # shuffle those bois
  train[[pair_name]] <- train[[pair_name]][sample(nrow(train[[pair_name]])),]
  test[[pair_name]] <- test[[pair_name]][sample(nrow(test[[pair_name]])),]
}

# # #
# Step 2: Machine learning funtime! I'll probably declare a few algorithms tbh
# # #

# Initialize ML pacakage h2o
h2o.init()

for (pair in word_pairs) {
  # # # Step 2.1 Initialize h2o dataframes / gridnames
  pair_name <- paste(pair, collapse="")

  this_train <- train[[pair_name]]
  this_test <- test[[pair_name]]

  train.hex <- as.h2o(this_train)
  test.hex <- as.h2o(this_test)

  # set all cols as factors for categorical analysis
  train.hex <- as.factor(train.hex)
  test.hex <- as.factor(test.hex)

  # # # Step 2.2 Commence GBM Classifier

  # save grid search
  grid_name <- paste0("gbm_", pair_name)

  # GBM hyper parameters
  params <- list(learn_rate = c(0.01, 0.05, 0.1),
                 max_depth = c(3, 10, 30))
  # Call grid search
  grid <- h2o.grid("gbm", y = "target",
                          grid_id = grid_name,
                          training_frame = train.hex,
                          validation_frame = test.hex,
                          learn_rate_annealing = 0.99,
                          ntrees = 100,
                          seed = 1,
                          hyper_params = params)

  grid_perf <- h2o.getGrid(grid_id = grid_name,
                           sort_by = "accuracy",
                           decreasing = T)

  # Choose top performing by accuracy
  m <- h2o.getModel(grid_perf@model_ids[[1]])

  # Write confusion matrix of test results to output file
  sink("gbm_results.txt", append = T)
  cat('\n')
  h2o.confusionMatrix(m, test.hex)
  sink(NULL)

  # # # End GBM Classifier

  # # # Step 2.3 Commence Neural Network

  # save grid search
  grid_name <- paste0("nn_", pair_name)

  # NN hyper parameters
  params <- list(hidden = c(1, 2, 5),
                 epochs = c(3, 5, 10))
  # Call grid search
  grid <- h2o.grid("deeplearning", y = "target",
                          grid_id = grid_name,
                          training_frame = train.hex,
                          validation_frame = test.hex,
                          seed = 1,
                          hyper_params = params)

  grid_perf <- h2o.getGrid(grid_id = grid_name,
                           sort_by = "accuracy",
                           decreasing = T)

  # Choose top performing by accuracy
  m <- h2o.getModel(grid_perf@model_ids[[1]])

  # Write confusion matrix of test results to output file
  sink("nn_results.txt", append = T)
  cat('\n')
  h2o.confusionMatrix(m, test.hex)
  sink(NULL)

  # # # End Neural Network Classifier
}


# Shutdown h2o
h2o.shutdown(prompt=T)
