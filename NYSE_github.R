library(tidyverse)
library(roperators)
library(magrittr)
library(plyr)
library(dplyr)
library(glmnet)
library(glmnetUtils)
library(readr)
library(tidyverse)
library(caret) 
library(margins)
library(janitor)
library('partykit')
library('ElemStatLearn')
library('randomForest')
library('randomForestExplainer')
library(leaps)
library(olsrr)

set.seed(11)

RMSE <-function(t, p) {
  sqrt(sum(((t-p)^2))*(1/nrow(t)))
}

# find dataset here ->> https://www.kaggle.com/camnugent/sandp500
price2<- read_csv("NYSE_pct_v3.csv")
fundamentals2 <- read_csv("fundamentals_pct_v3.csv")

#cleaning

DF_working <- inner_join(x =  price2,
                         y = fundamentals2,
                         by = c('date' = 'Period Ending', 'symbol' = 'Ticker Symbol')
)

DF_working2 <- tibble(change_dollar = DF_working$close/(DF_working$p_change+1),
                      log_change_dollar = log(change_dollar)
)

nyse <- bind_cols(DF_working2, DF_working)

#clean names
nyse_df<- clean_names(nyse)
nyse_df <- na.omit(nyse_df)
nyse_df <- nyse_df %>% select(- addl_income_expense_items, - cost_of_revenue, - goodwill, - gross_margin,
                              - net_cash_flows_investing, - net_income_adjustments, - operating_income,
                              - other_operating_items, - pre_tax_margin, - profit_margin, - short_term_investments,
                              - total_assets, - total_liabilities, - total_liabilities_equity)

trainSize <- .70
train_idx <-sample(1:nrow(nyse_df), size =floor(nrow(nyse_df)*
                                                  trainSize))

nyse_train<- nyse_df%>% slice(train_idx)
nyse_test <- nyse_df%>% slice(-train_idx)

mod1 <- lm(log_change_dollar ~ ., 
           data=subset(nyse_train, select=c( - change_dollar, - date, - symbol, - close, - p_change)))

collin <- ols_vif_tol(mod1)
glimpse(collin)

### Step Wise ###

#forward
fwd_fit_log <- regsubsets(log_change_dollar ~ .,
                          data = subset(nyse_train, select = c(- change_dollar, - date, - symbol, - close, - p_change)),
                          nvmax = 7,
                          method = "forward")
summary(fwd_fit_log)
plot(fwd_fit_log, scale = "adjr2")

scores_fwd_fit_log <- data.frame(
  scores = predict(rf_fit_log_100_sub,
                   newdata= nyse_train,
                   na.rm = TRUE,
                   type = "response"),
  nyse_train
) 

ggplot(data = scores_fwd_fit_log )+
  geom_point(mapping = aes(x = scores, y = log_change_dollar))+
  geom_abline(color = "red")

scores_fwd_fit_log_test <- data.frame(
  scores = predict(rf_fit_log_100_sub,
                   newdata= nyse_test,
                   na.rm = TRUE,
                   type = "response"),
  nyse_test
) 
ggplot(data = scores_fwd_fit_log_test )+
  geom_point(mapping = aes(x = scores, y = log_change_dollar))+
  geom_abline(color = "red")

#backward
bkwd_fit_log <- regsubsets(log_change_dollar ~ .,
                           data = subset(nyse_train, select = c(- change_dollar, - date, - symbol, - close, - p_change)),
                           nvmax = 7,
                           method = "backward")
summary(bkwd_fit_log)
plot(bkwd_fit_log)

### random forest ###

rf_fit_log <- randomForest(log_change_dollar ~ .,
                           data=subset(nyse_train, 
                                       select=c( - change_dollar, 
                                                 - date, 
                                                 - symbol, 
                                                 - close, 
                                                 - p_change)),
                           type = regression,
                           mtry = sqrt(64), 
                           ntree = 1000,
                           na.rm = TRUE,
                           importance = TRUE,
                           localImp = TRUE)

rf_fit_log

plot(rf_fit_log)

scores_rf1_log_train <- data.frame(
  scores = predict(rf_fit_log,
                   newdata= nyse_train,
                   na.rm = TRUE,
                   type = "response"),
  nyse_train
) 


ggplot(data = scores_rf1_log_train )+
  geom_point(mapping = aes(x = scores, y = log_change_dollar))+
  geom_abline(color = "red")


scores_rf1_log_test <- data.frame(
  scores = predict(rf_fit_log,
                   newdata= nyse_test,
                   na.rm = TRUE,
                   type = "response"),
  nyse_test
) 

ggplot(data = scores_rf1_log_test)+
  geom_point(mapping = aes(x = scores, y = log_change_dollar))+
  geom_abline(color = "red")


RMSE(scores_rf1_log_train %>% select(scores), scores_rf1_log_train %>% select(log_change_dollar))

RMSE(scores_rf1_log_test %>% select(scores), scores_rf1_log_test %>% select(log_change_dollar))

R2(scores_rf1_log_train$scores,scores_rf1_log_train$log_change_dollar)
R2(scores_rf1_log_test$scores,scores_rf1_log_test$log_change_dollar)

### Enet ###

alpha_grid <- seq(0,1, length = 100)

enet_mod_log <- cva.glmnet( log_change_dollar ~ ., 
                            data=subset(nyse_train, select=c( - change_dollar, - date, - symbol, - close, - p_change)), 
                            alpha = alpha_grid)

minlossplot(enet_mod_log)

enet_fit1_log <- cv.glmnet(log_change_dollar ~ .,
                           data=data_sub,
                           alpha = .0001)

plot(enet_fit1_log)

### LOOCV RF ###

preds_LOOCV <- rep(NA,nrow(nyse_train))

data_sub <- subset(nyse_train, select=c( - change_dollar, - date, - symbol, - close, - p_change))
num_rows <- nrow(data_sub)


for(i in 1:num_rows){
  mod <- randomForest(log_change_dollar ~ .,
                      data = data_sub %>% slice(-i),
                      type = regression,
                      mtry = sqrt(64),
                      ntree = 100,
                      na.rm = TRUE,
                      importance = TRUE,
                      localImp = TRUE)
  
  
  
  preds_LOOCV[i] <- predict(mod, newdata = data_sub %>% slice(i))
}

LOOCV_scores <- tibble(preds_LOOCV,
                       nyse_train)

RMSE(LOOCV_scores %>% select(preds_LOOCV), nyse_train %>% select(log_change_dollar))



preds_insample <- predict(rf_fit_log_100)
view(preds_insample)

insample <- tibble(preds_insample,
                   nyse_train)

view(insample)



RMSE(insample %>% select(preds_insample), nyse_train %>% select(log_change_dollar))



ggplot(data = LOOCV_scores)+
  geom_point(mapping = aes(x = preds_LOOCV, y = nyse_train$log_change_dollar))+
  geom_abline(color = "red")+
  ylab("actual")


ggplot(data = insample)+
  geom_point(mapping = aes(x = preds_insample, y = nyse_train$log_change_dollar))+
  geom_abline(color = "red")+
  ylab("actual")



#test
preds_LOOCV1 <- rep(NA,nrow(nyse_test))

data_sub1 <- subset(nyse_test, select=c( - change_dollar, - date, - symbol, - close, - p_change))
num_rows1 <- nrow(data_sub1)

for(i in 1:num_rows1){
  mod <- randomForest(log_change_dollar ~ .,
                      data = data_sub1 %>% slice(-i),
                      type = regression,
                      mtry = sqrt(78), 
                      ntree = 100,
                      na.rm = TRUE,
                      importance = TRUE,
                      localImp = TRUE)
  
  preds_LOOCV1[i] <- predict(mod, newdata = data_sub1 %>% slice(i))
}

LOOCV_scores1 <- tibble(preds_LOOCV1,
                        nyse_test)

view(LOOCV_scores1)

RMSE(LOOCV_scores1 %>% select(preds_LOOCV1), nyse_test %>% select( log_change_dollar))

is.numeric(LOOCV_scores1$nyse_test$log_change_dollar)

ggplot(data = LOOCV_scores1)+
  geom_point(mapping = aes(x = preds_LOOCV1, y = nyse_test$log_change_dollar))+
  geom_abline(color = "red")