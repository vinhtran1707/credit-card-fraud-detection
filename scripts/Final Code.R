rm(list=ls())
getwd()
setwd("C:/Users/MSI KATANA/Desktop/Tulane/Modeling and Analytics/Group Project")

getwd()


# Load required libraries
library(caret)
library(pROC)
library(gains)
library(rpart)
library(rpart.plot)
library(randomForest)

# ============================================================================
# PART 1: LOAD AND EXPLORE DATA
# ============================================================================

fraud.df <- read.csv("creditcard.csv")

dim(fraud.df)
head(fraud.df)
str(fraud.df)
summary(fraud.df)

# Check missing values
colSums(is.na(fraud.df))

# Class imbalance analysis
table(fraud.df$Class)
prop.table(table(fraud.df$Class)) * 100

# Visualize class distribution
barplot(table(fraud.df$Class), 
        names.arg = c("Legitimate", "Fraud"),
        main = "Class Distribution (Imbalanced: 0.17% Fraud)",
        col = c("steelblue", "salmon"))

# ============================================================================
# PART 2: FEATURE ANALYSIS
# ============================================================================

# Correlation with target
cor_with_class <- cor(fraud.df[, -31], fraud.df$Class)
cor_sorted <- sort(abs(cor_with_class[, 1]), decreasing = TRUE)
head(cor_sorted, 10)

# Top features visualization
top_features <- names(head(cor_sorted, 10))
barplot(cor_with_class[top_features, 1], 
        names.arg = top_features,
        main = "Top 10 Features Correlated with Fraud",
        col = ifelse(cor_with_class[top_features, 1] > 0, "steelblue", "salmon"),
        las = 2)

# Amount by class
boxplot(Amount ~ Class, data = fraud.df,
        main = "Transaction Amount by Class",
        col = c("steelblue", "salmon"))

# ============================================================================
# PART 3: FEATURE SCALING AND DATA PREPARATION
# ============================================================================

# Scale Time and Amount (V1-V28 already PCA-transformed)
fraud.df.scaled <- fraud.df
fraud.df.scaled$Time <- scale(fraud.df$Time)[, 1]
fraud.df.scaled$Amount <- scale(fraud.df$Amount)[, 1]

# Convert Class to factor
fraud.df.scaled$Class <- as.factor(fraud.df.scaled$Class)

# ============================================================================
# PART 4: STRATIFIED TRAIN/VALIDATION SPLIT (80/20)
# ============================================================================

set.seed(123)
train.index <- createDataPartition(fraud.df.scaled$Class, p = 0.8, list = FALSE)

train.df <- fraud.df.scaled[train.index, ]
valid.df <- fraud.df.scaled[-train.index, ]

# Verify split
dim(train.df)
dim(valid.df)
table(train.df$Class)
table(valid.df$Class)

# ============================================================================
# PART 5: BALANCE TRAINING DATA (FOR MODELS 2 & 3)
# ============================================================================

# Function to balance data via combined over/undersampling
manual_balance <- function(data, target_col = "Class", ratio = 0.5) {
  majority <- data[data[[target_col]] == "0", ]
  minority <- data[data[[target_col]] == "1", ]
  
  n_minority <- nrow(minority)
  target_majority <- round(n_minority / ratio - n_minority)
  
  minority_oversampled <- minority[sample(nrow(minority), n_minority, replace = TRUE), ]
  majority_undersampled <- majority[sample(nrow(majority), target_majority, replace = FALSE), ]
  
  balanced <- rbind(majority_undersampled, minority_oversampled)
  return(balanced[sample(nrow(balanced)), ])
}

train.balanced <- manual_balance(train.df, ratio = 0.5)
table(train.balanced$Class)

# ============================================================================
# PART 6: MODEL BUILDING
# ============================================================================

# ------------------------------------------------------------------------------
# MODEL 1: BASELINE LOGISTIC REGRESSION (IMBALANCED DATA)
# ------------------------------------------------------------------------------

logit.baseline <- glm(Class ~ ., data = train.df, family = "binomial")
logit.baseline.pred <- predict(logit.baseline, valid.df[, -31], type="response")

cm1 <- confusionMatrix(as.factor(ifelse(logit.baseline.pred >= 0.5, 1, 0)), 
                       valid.df$Class, positive = "1")
cm1

# ------------------------------------------------------------------------------
# MODEL 2: LOGISTIC REGRESSION (BALANCED DATA)
# ------------------------------------------------------------------------------

logit.balanced <- glm(Class ~ ., data = train.balanced, family = "binomial")
logit.balanced.pred <- predict(logit.balanced, valid.df[, -31], type="response")

cm2 <- confusionMatrix(as.factor(ifelse(logit.balanced.pred >= 0.5, 1, 0)), 
                       valid.df$Class, positive = "1")
cm2

# ROC Curve
roc2 <- roc(valid.df$Class, logit.balanced.pred)
plot.roc(roc2, main = "ROC Curve - Logistic Regression (Balanced)")
auc(roc2)

# Lift Chart
gain2 <- gains(as.numeric(as.character(valid.df$Class)), logit.balanced.pred, groups=10)
plot(c(0, gain2$cume.pct.of.total * sum(as.numeric(as.character(valid.df$Class)))) ~ 
       c(0, gain2$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="Lift Chart - LR Balanced", type="l")
lines(c(0, sum(as.numeric(as.character(valid.df$Class))))~c(0, nrow(valid.df)), lty=2)

# ------------------------------------------------------------------------------
# MODEL 3: DECISION TREE (BALANCED DATA WITH CV)
# ------------------------------------------------------------------------------

set.seed(123)
tree.cv <- rpart(Class ~ ., data = train.balanced, method = "class", 
                 cp = 0.00001, minsplit = 5, xval = 5)

printcp(tree.cv)
tree.pruned <- prune(tree.cv, cp = 0.0001)

rpart.plot(tree.pruned, type=4, extra=1, under=TRUE, 
           main = "Decision Tree for Fraud Detection")

tree.pred <- predict(tree.pruned, valid.df, type = "class")
cm3 <- confusionMatrix(tree.pred, valid.df$Class, positive = "1")
cm3

# ------------------------------------------------------------------------------
# MODEL 4: RANDOM FOREST (CLASS WEIGHTS - PROPERLY CONFIGURED)
# ------------------------------------------------------------------------------

# Use class weights instead of balancing
class_weights <- c("0" = 1, "1" = 577)

set.seed(123)
rf.weighted <- randomForest(Class ~ ., 
                            data = train.df,
                            ntree = 100,
                            mtry = 5,
                            nodesize = 1,           # FIXED: Default for better learning
                            classwt = class_weights,
                            importance = TRUE)
# NOTE: Removed maxnodes - it was causing severe underfitting!

# Get probability predictions
rf.pred.prob <- predict(rf.weighted, valid.df, type = "prob")[, 2]

# Variable importance
varImpPlot(rf.weighted, main = "Variable Importance - Random Forest")

# ============================================================================
# PART 7: INITIAL MODEL COMPARISON (before RF optimization)
# ============================================================================

# Evaluate RF at default 0.5 threshold for initial comparison
rf.pred.default <- ifelse(rf.pred.prob >= 0.5, 1, 0)
cm4_default <- confusionMatrix(as.factor(rf.pred.default), valid.df$Class, positive = "1")

comparison_initial <- data.frame(
  Model = c("LR_Baseline", "LR_Balanced", "Tree_CV", "RF_Weighted_0.5"),
  Accuracy = c(cm1$overall[1], cm2$overall[1], cm3$overall[1], cm4_default$overall[1]),
  Sensitivity = c(cm1$byClass[1], cm2$byClass[1], cm3$byClass[1], cm4_default$byClass[1]),
  Specificity = c(cm1$byClass[2], cm2$byClass[2], cm3$byClass[2], cm4_default$byClass[2]),
  Precision = c(cm1$byClass[3], cm2$byClass[3], cm3$byClass[3], cm4_default$byClass[3]),
  F1 = c(cm1$byClass[7], cm2$byClass[7], cm3$byClass[7], cm4_default$byClass[7])
)

comparison_initial

# ============================================================================
# PART 8: THRESHOLD OPTIMIZATION FOR RANDOM FOREST
# ============================================================================

# Test different thresholds
thresholds <- seq(0.10, 0.90, by = 0.05)
threshold_results <- data.frame()

for (thresh in thresholds) {
  pred <- ifelse(rf.pred.prob >= thresh, 1, 0)
  cm_temp <- confusionMatrix(as.factor(pred), valid.df$Class, positive = "1")
  
  threshold_results <- rbind(threshold_results, 
                             data.frame(Threshold = thresh,
                                        Precision = cm_temp$byClass[3],
                                        Recall = cm_temp$byClass[1],
                                        F1 = cm_temp$byClass[7],
                                        Specificity = cm_temp$byClass[2]))
}

threshold_results

# Find optimal threshold
optimal_thresh <- threshold_results$Threshold[which.max(threshold_results$F1)]
optimal_thresh

# Visualize threshold impact
plot(threshold_results$Threshold, threshold_results$Recall, 
     type = "l", col = "red", lwd = 2, ylim = c(0, 1),
     xlab = "Threshold", ylab = "Metric",
     main = "Threshold Optimization for Random Forest")
lines(threshold_results$Threshold, threshold_results$Precision, col = "blue", lwd = 2)
lines(threshold_results$Threshold, threshold_results$F1, col = "purple", lwd = 2)
abline(v = optimal_thresh, lty = 2, col = "green", lwd = 2)
legend("right", legend = c("Recall", "Precision", "F1", "Optimal"), 
       col = c("red", "blue", "purple", "green"), lwd = 2)

# ============================================================================
# PART 9: FINAL MODEL PERFORMANCE WITH OPTIMAL THRESHOLD
# ============================================================================

# Apply optimal threshold
rf.pred.final <- ifelse(rf.pred.prob >= optimal_thresh, 1, 0)
cm.final <- confusionMatrix(as.factor(rf.pred.final), valid.df$Class, positive = "1")
cm.final

# Business cost analysis - CORRECTED INDEXING
# Confusion matrix structure:
#              Reference
# Prediction     0     1
#          0    TN    FN  (Predicted negative)
#          1    FP    TP  (Predicted positive)

tn <- cm.final$table[1, 1]  # True Negatives: Predicted 0, Actually 0
fn <- cm.final$table[1, 2]  # False Negatives: Predicted 0, Actually 1 (MISSED FRAUDS)
fp <- cm.final$table[2, 1]  # False Positives: Predicted 1, Actually 0 (FALSE ALARMS)
tp <- cm.final$table[2, 2]  # True Positives: Predicted 1, Actually 1 (CAUGHT FRAUDS)

cost_per_fraud <- 500
cost_per_false_alarm <- 50

# Calculate business metrics
savings_detected <- tp * cost_per_fraud         # Money saved from catching frauds
losses_missed <- fn * cost_per_fraud            # Money lost from missed frauds
cost_false_alarms <- fp * cost_per_false_alarm  # Cost of investigating false alarms

net_benefit <- savings_detected - losses_missed - cost_false_alarms
fraud_detection_rate <- tp / (tp + fn)
false_positive_rate <- fp / (tn + fp)

# Display results
print("=== BUSINESS METRICS ===")
savings_detected
losses_missed
cost_false_alarms
net_benefit
fraud_detection_rate
false_positive_rate

# ============================================================================
# FINAL COMPARISON: ALL MODELS WITH RF OPTIMIZED
# ============================================================================

comparison_final <- data.frame(
  Model = c("LR_Baseline", "LR_Balanced", "Tree_CV", "RF_Weighted_Optimized"),
  Accuracy = c(cm1$overall[1], cm2$overall[1], cm3$overall[1], cm.final$overall[1]),
  Sensitivity = c(cm1$byClass[1], cm2$byClass[1], cm3$byClass[1], cm.final$byClass[1]),
  Specificity = c(cm1$byClass[2], cm2$byClass[2], cm3$byClass[2], cm.final$byClass[2]),
  Precision = c(cm1$byClass[3], cm2$byClass[3], cm3$byClass[3], cm.final$byClass[3]),
  F1 = c(cm1$byClass[7], cm2$byClass[7], cm3$byClass[7], cm.final$byClass[7])
)

comparison_final

# Best model by F1-Score
comparison_final[which.max(comparison_final$F1), ]

# Annual projection
annual_benefit <- net_benefit / 2 * 365
annual_benefit

# ============================================================================
# SUMMARY
# ============================================================================

print("=== PROJECT SUMMARY ===")
print(paste("Best Model: Random Forest with Class Weights"))
print(paste("Optimal Threshold:", optimal_thresh))
print(paste("F1-Score:", round(cm.final$byClass[7], 4)))
print(paste("Recall:", round(cm.final$byClass[1], 4)))
print(paste("Precision:", round(cm.final$byClass[3], 4)))
print(paste("Frauds Detected:", tp, "out of", tp+fn))
print(paste("False Alarms:", fp))
print(paste("Net Benefit (2 days): €", net_benefit))
print(paste("Annual Projection: €", round(annual_benefit)))
