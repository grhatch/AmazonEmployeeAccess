## Amazon Analysis
library(tidyverse)
library(vroom)
library(tidymodels)
library(stacks)
library(poissonreg)
library(ggmosaic)
library(embed)
library(ggplot2)


# Read in data
train <- vroom("./STAT348/AmazonEmployeeAccess/amazon-employee-access-challenge/train.csv")
test <- vroom("./STAT348/AmazonEmployeeAccess/amazon-employee-access-challenge/test.csv")


eda_data <- data.frame(lapply(train, factor))

#BAKE THE COLUMNS IN THE RECIPE

eda_recipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors())  # dummy variable encoding
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  #step_mutate_at(all_numeric_predictors(), fn = factor)
  

prep <- prep(eda_recipe)
baked <- bake(prep, new_data = train)

eda_data <- data.frame(lapply(baked, factor))


ggplot(data=eda_data) + 
  geom_mosaic(aes(x=product(RESOURCE_other), fill=ACTION))

ggplot(data=eda_data) + 
  geom_boxplot(aes(x=ROLE_FAMILY_DESC_other, y=ACTION))



###########################
## Logistic Regression ####
###########################

train_new <- train %>%
  mutate(ACTION = as.factor(ACTION))

logreg_recipe <- recipe(ACTION~., data=train_new) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) # dummy variable encoding
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  #step_mutate_at(all_numeric_predictors(), fn = factor)

logreg_model <- logistic_reg() %>% #Type of model
  set_engine("glm")

logreg_workflow <- workflow() %>%
  add_recipe(logreg_recipe) %>%
  add_model(logreg_model) %>%
  fit(data = train_new) # Fit the workflow

amazon_pred <- predict(logreg_workflow,
                              new_data=test,
                              type="prob") %>% # "class" or "prob" (see doc
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_1) %>% # Just keep datetime and predictions
  rename(Action = .pred_1) # rename pred to count (for submission to Kaggle)
  #mutate(Action = pmax(0,Action)) %>% #pointwise max of (0,prediction)
  #mutate(Action = ifelse(is.na(Action), 0, Action)) %>%
  #mutate(datetime=as.character(format(datetime))) #needed for right format

vroom_write(amazon_pred, "amazon_pred.csv", delim = ',')



###################################
## Penalized Logistic Regression ##
###################################

penlog_recipe <- recipe(ACTION~., data=train_new) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding

penlog_model <- logistic_reg(mixture=tune(),penalty=tune()) %>%
  set_engine("glmnet")

penlog_wf <- workflow() %>%
  add_recipe(penlog_recipe) %>%
  add_model(penlog_model)

L <- 5
## Grid of values to tune over
penlog_tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
folds <- vfold_cv(train_new, v = K, repeats=1)

## Run the CV
penlog_CV_results <- penlog_wf %>%
  tune_grid(resamples=folds,
          grid=penlog_tuning_grid,
          metrics=metric_set(roc_auc
                             #, f_meas, sens, recall, spec,precision, accuracy
                             )) #Or leave metrics NULL


## Find Best Tuning Parameters
penlog_bestTune <- penlog_CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
penlog_final_wf <- penlog_wf %>%
  finalize_workflow(penlog_bestTune) %>%
  fit(data=train_new)

## Predict
penlog_pred <- penlog_final_wf %>%
  predict(new_data = test, type="prob") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_1) %>% # Just keep datetime and predictions
  rename(Action = .pred_1) # rename pred to count (for submission to Kaggle)

vroom_write(penlog_pred, "penlog_pred.csv", delim = ',')
