# Install glmnet if you haven't already: install.packages("glmnet")
library(glmnet)

#' Generate Lasso Prediction Intervals using Residual Bootstrap
#'
#' @param X_train The predictor matrix for training
#' @param y_train The response vector for training
#' @param X_new The predictor matrix for new observations
#' @param B Number of bootstrap iterations (default 500)
#' @param alpha Significance level for the interval (default 0.05 for 95% PI)
#' @return A data frame containing the point prediction, lower bound, and upper bound.
lasso_pred_interval <- function(X_train, y_train, X_new, B = 500, alpha = 0.05) {
  
  n <- nrow(X_train)
  n_new <- nrow(X_new)
  
  # 1. Fit CV model to find the optimal penalty (lambda)
  cv_fit <- cv.glmnet(X_train, y_train, alpha = 1)
  best_lambda <- cv_fit$lambda.min
  
  # 2. Get baseline predictions and residuals using the full data
  fit_orig <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)
  preds_orig <- predict(fit_orig, newx = X_train)
  residuals <- y_train - preds_orig
  
  # Centering residuals is a best practice in bootstrapping
  residuals <- residuals - mean(residuals) 
  
  # Matrix to store bootstrap predictions for the new data
  boot_preds <- matrix(NA, nrow = n_new, ncol = B)
  
  # 3. Bootstrap loop
  for (i in 1:B) {
    # Resample residuals with replacement
    boot_res <- sample(residuals, size = n, replace = TRUE)
    
    # Create bootstrap response
    y_boot <- preds_orig + boot_res
    
    # Refit the Lasso model (keeping lambda fixed to the optimal one for speed/stability)
    boot_fit <- glmnet(X_train, y_boot, alpha = 1, lambda = best_lambda)
    
    # Predict on the new data
    point_preds <- predict(boot_fit, newx = X_new)
    
    # Add a resampled residual to account for the prediction error variance
    new_error <- sample(residuals, size = n_new, replace = TRUE)
    boot_preds[, i] <- point_preds + new_error
  }
  
  # 4. Calculate quantiles for the intervals
  lower_bound <- apply(boot_preds, 1, quantile, probs = alpha / 2)
  upper_bound <- apply(boot_preds, 1, quantile, probs = 1 - (alpha / 2))
  
  # Base point predictions (without bootstrap noise)
  point_estimate <- predict(fit_orig, newx = X_new)
  
  # Return results as a data frame
  results <- data.frame(
    Point_Prediction = as.vector(point_estimate),
    Lower_PI = lower_bound,
    Upper_PI = upper_bound
  )
  
  return(results)
}

# ==========================================
# Example Usage
# ==========================================

# Generate synthetic training data
set.seed(42)
n_train <- 150
p <- 10
X_train <- matrix(rnorm(n_train * p), n_train, p)
# True model relies only on the first two predictors
y_train <- X_train[, 1] * 5 + X_train[, 2] * -3 + rnorm(n_train, sd = 2)

# Generate new data we want to predict
X_new <- matrix(rnorm(3 * p), 3, p)

# Get the 95% prediction intervals
predictions <- lasso_pred_interval(
  X_train = X_train, 
  y_train = y_train, 
  X_new = X_new, 
  B = 500,        # Increase B to 1000+ for production environments
  alpha = 0.05    # 95% Interval
)

print("Predictions and 95% Prediction Intervals:")
print(predictions)