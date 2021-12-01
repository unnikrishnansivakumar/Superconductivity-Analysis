library(ggplot2)

supercon_data <- read.csv('dataset/train.csv')

head(supercon_data,5)

dim(supercon_data)

sum(is.na(supercon_data))

table(supercon_data$number_of_elements)

table(supercon_data$range_Valence)

ggplot(data=supercon_data, aes(supercon_data$critical_temp)) + 
  geom_histogram(col="darkblue", 
                 fill="darkblue", 
                 alpha=.4)

colNames <- names(supercon_data)

for (i in colNames){
    ggplot(supercon_data, aes_string(x=i,y='critical_temp'))+geom_point()
    ggsave(paste("plots/",i,".jpg",sep=""))  
}

lev_fit <- lm(critical_temp ~ ., data = supercon_data)
# Use function "hatvalues" for the leverages from the lm object
leverages = hatvalues(lev_fit)
max_i <- which.max(leverages)

par(mfrow=c(1,2))
plot(fitted(lev_fit), resid(lev_fit), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Fitted versus Residuals")
abline(h = 0, col = "darkorange", lwd = 2)

qqnorm(resid(lev_fit), col = "grey",pch=20,cex=2)
qqline(resid(lev_fit), col = "dodgerblue", lwd = 2)

lev_fit_cd <- cooks.distance(lev_fit)

sum(lev_fit_cd > 4/length(lev_fit_cd))

inf_i = which(lev_fit_cd > 4/length(lev_fit_cd))
supercon_data_ou_tr = supercon_data[-inf_i,]

lev_fit_2 = lm(critical_temp ~ ., data = supercon_data_ou_tr)

par(mfrow=c(1,2))
plot(fitted(lev_fit_2), resid(lev_fit_2), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Fitted versus Residuals")
abline(h = 0, col = "darkorange", lwd = 2)

qqnorm(resid(lev_fit_2), col = "grey",pch=20,cex=2)
qqline(resid(lev_fit_2), col = "dodgerblue", lwd = 2)

supercon_data = supercon_data_ou_tr

library(corrplot)

library(dplyr)
library(caret)

# Correlation plot - High Correlation among variables
# It can be seen that high correlation exists!
correlations <- cor(supercon_data)
corrplot(correlations, method="circle",tl.cex=0.5)

cor_matrix = cor(supercon_data)

drop = findCorrelation(cor_matrix, cutoff = .75) #function that returns a vector of integers corresponding to columns to remove to reduce pair-wise correlations.
drop = names(supercon_data)[drop]
supercon_data_corr_rem = supercon_data[ , !(names(supercon_data) %in% drop)]

correlations <- cor(supercon_data_corr_rem)
corrplot(correlations, method="circle",tl.cex=0.5)

#Getting the highly correlated variable groups for better explainability
row_col_mat = which(cor_matrix>=0.6, arr.ind=TRUE)
rc_df = as.data.frame(row_col_mat)
correlated_var_groups = rc_df %>% group_by(row) %>% 
summarize(col = paste(sort(unique(col)),collapse=", "))

smp_size <- floor(0.70 * nrow(supercon_data_corr_rem))

## set the seed to make your partition reproducible
set.seed(10)
train_ind <- sample(seq_len(nrow(supercon_data_corr_rem)), size = smp_size)

train <- supercon_data_corr_rem[train_ind, ]

test <- supercon_data_corr_rem[-train_ind, ]

model_correlated_rem<-lm(critical_temp~.,data=train)
summary(model_correlated_rem)

qqnorm(resid(model_correlated_rem), col = "grey",pch=20,cex=2)
qqline(resid(model_correlated_rem), col = "dodgerblue", lwd = 2)

par(mfrow=c(1,2))
plot(fitted(model_correlated_rem), resid(model_correlated_rem), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Fitted versus Residuals")
abline(h = 0, col = "darkorange", lwd = 2)

model_taylor_tf<-lm(sqrt(critical_temp)~.,data=train)
summary(model_taylor_tf)

par(mfrow=c(1,2))
plot(fitted(model_taylor_tf), resid(model_taylor_tf), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Fitted versus Residuals")
abline(h = 0, col = "darkorange", lwd = 2)

qqnorm(resid(model_taylor_tf), col = "grey",pch=20,cex=2)
qqline(resid(model_taylor_tf), col = "dodgerblue", lwd = 2)

boxcox(model_correlated_rem)

boxcox(model_correlated_rem, lambda = seq(-0, 0.5, by = 0.01))

lambda = 0.29
model_boxcox_supercon <- lm(((critical_temp^(lambda)-1)/(lambda))~.,data=train)

summary(model_boxcox_supercon)

par(mfrow=c(1,2))
plot(fitted(model_boxcox_supercon), resid(model_boxcox_supercon), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Fitted versus Residuals")
abline(h = 0, col = "darkorange", lwd = 2)

qqnorm(resid(model_boxcox_supercon), col = "grey",pch=20,cex=2)
qqline(resid(model_boxcox_supercon), col = "dodgerblue", lwd = 2)

library(ISLR)
library(glmnet)

X_train = model.matrix(critical_temp~. ,train)[, -1]# the last column (for intercept) is eliminated
y_train = train$critical_temp
X_test = model.matrix(critical_temp~. ,test)[, -1]
y_test = test$critical_temp

eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square,
    adj.r.squared=1- ((1-R_square)*(21263-1)/(21263-27-1))
  )
  
}

cv_model_regularization = cv.glmnet(X_train, y_train, alpha = 1)
plot(cv_model_regularization)

bestlam = cv_model_regularization$lambda.min
bestlam

model_regularization <- glmnet(X_train, y_train, alpha = 1, lambda = 0.02)


ypr4 = predict(model_regularization, X_test, lambda = 0.02)
eval_results(y_test, ypr4, test)

lasso_coef = coef(model_regularization)
sum(lasso_coef != 0)

adj.r.squared=

resid_reg = ypr4-y_test
fitted_reg = ypr4

qqnorm(resid_reg, col = "grey",pch=20,cex=2)
qqline(resid_reg, col = "dodgerblue", lwd = 2)

par(mfrow=c(1,2))
plot(fitted_reg, resid_reg, col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Fitted versus Residuals")
abline(h = 0, col = "darkorange", lwd = 2)

coef(model_regularization)

plot(ypr4,y_test,xlab="predicted",ylab="actual")
 abline(a=0,b=1,col='red',lwd=2)

#construct full model
model_full = lm(critical_temp~.,data=supercon_data)

fit_back_aic = step(model_full, direction = "backward")
fit_back_aic

n = nrow(supercon_data)
fit_back_bic = step(model_full, direction = "backward", k=log(n))
fit_back_bic

summary(fit_back_aic)

summary(fit_back_bic)
