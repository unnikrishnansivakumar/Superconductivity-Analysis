



supercon_data =read.csv('train.csv')

smp_size <- floor(0.70 * nrow(supercon_data))

## set the seed to make your partition reproducible
set.seed(10)
train_ind <- sample(seq_len(nrow(supercon_data)), size = smp_size)

train <- supercon_data[train_ind, ]

test <- supercon_data[-train_ind, ]


X_train = model.matrix(critical_temp~. ,train)[, -1]# the last column (for intercept) is eliminated
y_train = train$critical_temp
X_test = model.matrix(critical_temp~. ,test)[, -1]
y_test = test$critical_temp


library(ISLR)

library(glmnet)



eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}


library(glmnet)







best_model <- glmnet(X_train, y_train, alpha = 1, lambda = 0.5)


predictions_train <- predict(best_model, s = 0.5, newx = X_train)
eval_results(y_train, predictions_train, train)

ypr4 = predict(best_model, X_test, lambda = 0.5)
eval_results(y_test, ypr4, test)

lasso_coef = coef(best_model)
sum(lasso_coef != 0)











