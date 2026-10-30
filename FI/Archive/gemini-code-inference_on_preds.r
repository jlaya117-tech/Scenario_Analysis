# Load required packages
# install.packages(c("glmnet", "selectiveInference"))
library(glmnet)
library(selectiveInference)

# 1. Simulate a quantitative dataset 
set.seed(42)
n <- 200  # Number of observations (e.g., monthly returns)
p <- 50   # Number of potential factors

X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("Factor_", 1:p)
y <- rnorm(n) # Asset returns (pure noise in this simulation)

# 2. Fit LASSO and find the optimal penalty using cross-validation
cv_fit <- cv.glmnet(X, y, standardize = TRUE)
opt_lambda <- cv_fit$lambda.1se # using 1se gives a sparser, more interpretable model

# 3. Fit the final LASSO model to extract the non-zero coefficients
lasso_fit <- glmnet(X, y, standardize = TRUE)
beta_hat <- coef(lasso_fit, s = opt_lambda, exact = TRUE, x = X, y = y)[-1]
active_set <- which(beta_hat != 0)

# 4. Post-Selection Inference
if(length(active_set) > 0) {
  
  # fixedLassoInf automatically calculates BOTH p-values and confidence intervals
  # Note: The function requires lambda scaled by n
  out <- fixedLassoInf(x = X, 
                       y = y, 
                       beta = beta_hat, 
                       lambda = opt_lambda * n, 
                       family = "gaussian",
                       alpha = 0.05) # Sets the level for 95% confidence intervals
  
  # 5. Compile the results into a clean dataframe
  results <- data.frame(
    Factor = colnames(X)[active_set],
    Lasso_Estimate = out$coef0,
    P_Value = out$pv,
    Lower_CI = out$ci[,1],
    Upper_CI = out$ci[,2]
  )
  
  print("Post-Selection Inference Results:")
  print(results)
  
} else {
  print("LASSO penalized all factors to zero. No features to test.")
}