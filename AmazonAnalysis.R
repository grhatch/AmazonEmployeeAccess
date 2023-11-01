## Amazon Analysis
library(tidyverse)
library(vroom)
library(tidymodels)
library(stacks)
library(poissonreg)
library(ggmosaic)
library(embed)
library(ggplot2)

# R - get into an interactive environment
# exit - leave server
# R CMD BATCH --no-save --no-restore AmazonAnalysis.R & - run entire r script in the background (&) without saving workspace and without loading something into the environment
# top - show running processes
# save(file="filename.RData", list=c("logReg_wf))
# load("filename.RData")




# Read in data
#train <- vroom("./STAT348/AmazonEmployeeAccess/amazon-employee-access-challenge/train.csv")
#test <- vroom("./STAT348/AmazonEmployeeAccess/amazon-employee-access-challenge/test.csv")

train <- vroom("./amazon-employee-access-challenge/train.csv") 
test <- vroom("./amazon-employee-access-challenge/test.csv")


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

######## SMOTE #################
library(themis) # for smote

balanced_recipe <- recipe(ACTION~., data=train_new) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
  step_smote(all_outcomes(), neighbors=4) %>%
  step_upsample() # or step_downsample()
##################################


logreg_recipe <- recipe(ACTION~., data=train_new) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) # dummy variable encoding
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  #step_mutate_at(all_numeric_predictors(), fn = factor)

logreg_model <- logistic_reg() %>% #Type of model
  set_engine("glm")

logreg_workflow <- workflow() %>%
  add_recipe(balanced_recipe) %>%
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

vroom_write(amazon_pred, "amazon_pred_smote.csv", delim = ',')



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
  add_recipe(balanced_recipe) %>%
  add_model(penlog_model)

L <- 10
## Grid of values to tune over
penlog_tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = L) ## L^2 total tuning possibilities

K <- 10
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

vroom_write(penlog_pred, "penlog_pred_smote_local.csv", delim = ',')




###################
## Random Forest ##
###################

rf_model <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")



rf_recipe <- recipe(ACTION~., data=train_new) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
  step_normalize(all_numeric_predictors())

# set up workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

L <- 10
## Grid of values to tune over; these should be params in the model
rf_tuning_grid <- grid_regular(mtry(range = c(1,1)),
                               min_n(),
                               levels = L) ## L^2 total tuning possibilities

K <- 10
## Split data for CV
rf_folds <- vfold_cv(train_new, v = K, repeats=1)

## Run CV
rf_CV_results <- rf_wf %>%
  tune_grid(resamples=rf_folds,
            grid=rf_tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
rf_bestTune <- rf_CV_results %>%
  select_best("roc_auc")


## Finalize the Workflow & fit it
rf_final_wf <-
  rf_wf %>%
  finalize_workflow(rf_bestTune) %>%
  fit(data=train_new)

## Predict
rf_pred <- rf_final_wf %>%
  predict(new_data = test, type="prob") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_1) %>% # Just keep datetime and predictions
  rename(Action = .pred_1) # rename pred to count (for submission to Kaggle)

vroom_write(rf_pred, "rf_classification_pred_smote_final.csv", delim = ',')


############################
## Naive Bayes Classifier ##
############################
library(discrim)

nb_recipe <- recipe(ACTION~., data=train_new) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


nb_wf <- workflow() %>%
  add_recipe(balanced_recipe) %>% ## switch this between nb_recipe and pcdr_recipe
  add_model(nb_model)

L <- 5
## Grid of values to tune over; these should be params in the model
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
nb_folds <- vfold_cv(train_new, v = K, repeats=1)

## Run CV
nb_CV_results <- nb_wf %>%
  tune_grid(resamples=nb_folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
nb_bestTune <- nb_CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
nb_final_wf <-
  nb_wf %>%
  finalize_workflow(nb_bestTune) %>%
  fit(data=train_new)

## Predict
nb_pred <- nb_final_wf %>%
  predict(new_data = test, type="prob") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_1) %>% # Just keep datetime and predictions
  rename(Action = .pred_1) # rename pred to count (for submission to Kaggle)

vroom_write(nb_pred, "nb_pcdr_classification_pred_smote.csv", delim = ',')



#########################
## K-Nearest-Neighbors ##
#########################

# make sure to normalize in the recipe

knn_recipe <- recipe(ACTION~., data=train_new) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding

# principal component dimension reduction option
pcdr_recipe <- recipe(ACTION~., data=train_new) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding
  step_pca(all_predictors(), threshold = .8) #threshhold of 1 means no dimension reduction



## knn mode
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(balanced_recipe) %>% ## switch this between knn_recipe and pcdr_recipe
  add_model(knn_model)

L <- 5
## Grid of values to tune over; these should be params in the model
knn_tuning_grid <- grid_regular(neighbors(),
                               levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
knn_folds <- vfold_cv(train_new, v = K, repeats=1)

## Run CV
knn_CV_results <- knn_wf %>%
  tune_grid(resamples=knn_folds,
            grid=knn_tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
knn_bestTune <- knn_CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
knn_final_wf <-
  knn_wf %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data=train_new)

## Predict
knn_pred <- knn_final_wf %>% 
  predict(new_data = test, type="prob") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_1) %>% # Just keep datetime and predictions
  rename(Action = .pred_1) # rename pred to count (for submission to Kaggle)

vroom_write(knn_pred, "pcdr_pred_smote.csv", delim = ',')


#############################
## Support Vector Machines ##
#############################

# radial kernel type is the best
# cost: penalty of classifying incorrectly
# rbf_sigma (radial kernels only) > 0: wigglyness of boundary (lower = wiggly, higher = linear)
# degree (polynomial kernels only) > 0: wigglyness of oundary (lower = linear, higher = wiggly)
# Categorical varibales need to be converted to numeric (step_dummy or target encoding)

svm_recipe <- recipe(ACTION~., data=train_new) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding
  
svmLinear_model <- svm_linear(cost = tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(balanced_recipe) %>% ## switch this between knn_recipe and pcdr_recipe
  add_model(svmLinear_model)

L <- 5
## Grid of values to tune over; these should be params in the model
svm_tuning_grid <- grid_regular(cost(),
                                levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
svm_folds <- vfold_cv(train_new, v = K, repeats=1)

## Run CV
svm_CV_results <- svm_wf %>%
  tune_grid(resamples=svm_folds,
            grid=svm_tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
svm_bestTune <- svm_CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
svm_final_wf <-
  svm_wf %>%
  finalize_workflow(svm_bestTune) %>%
  fit(data=train_new)

## Predict
svm_pred <- svm_final_wf %>% 
  predict(new_data = test, type="prob") %>%
  bind_cols(.,test) %>% # bind predictions with test data
  select(id, .pred_1) %>% # Just keep datetime and predictions
  rename(Action = .pred_1) # rename pred to count (for submission to Kaggle)

vroom_write(svm_pred, "svm_pred_smote.csv", delim = ',')

  
  

####################
## Imbalaned Data ##
####################

# sometimes we want to predict something with very unbalanced data 
# (i.e. detecting fraud; the vast majority of transactions are not fraudulant)
# Majority Class = class w highest num obs
# Minority Class = opposite
