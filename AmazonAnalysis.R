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
