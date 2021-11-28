



supercon_data =read.csv('train.csv')

smp_size <- floor(0.75 * nrow(supercon_data))

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

#perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(X_train, y_train, alpha = 1,)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda



#produce plot of test MSE by lambda value



best_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)


predictions_train <- predict(best_model, s = best_lambda, newx = X_train)
eval_results(y_train, predictions_train, train)

ypr4 = predict(best_model, X_test, lambda = best_lambda)
eval_results(y_test, ypr4, test)

lasso_coef = coef(best_model)
sum(lasso_coef != 0)









fit_ridge_cv= cv.glmnet(X_train, y_train, alpha = 0)# choosing best lambda using CV
bestlam = fit_ridge_cv$lambda.min
fit_ridge_best_cv = glmnet(X_train, y_train, alpha = 0, lambda = bestlam)

ypr3 = predict(fit_ridge_best_cv, X_test, lambda = bestlam)
eval_results(y_test, ypr3, test)

lasso_coef = coef(fit_ridge_best_cv)







