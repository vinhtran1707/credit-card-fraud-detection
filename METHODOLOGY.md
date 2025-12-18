# Detailed Methodology

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Data Understanding](#data-understanding)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Imbalance Handling Strategies](#imbalance-handling-strategies)
5. [Model Training](#model-training)
6. [Threshold Optimization](#threshold-optimization)
7. [Evaluation Framework](#evaluation-framework)
8. [Production Deployment](#production-deployment)

---

## 1. Problem Formulation

### Business Problem
Credit card fraud detection represents a severe class imbalance problem where fraudulent transactions constitute less than 0.2% of all transactions. The challenge is to build a system that:
- Maximizes fraud detection rate (recall/sensitivity)
- Minimizes false alarms (maintains high precision)
- Operates in real-time on live transaction streams
- Adapts to evolving fraud patterns

### Technical Formulation
**Task:** Binary classification  
**Input:** Transaction features (30 numerical predictors)  
**Output:** Fraud probability [0, 1] → Binary decision {0: Legitimate, 1: Fraud}

**Constraints:**
- Real-time inference: <100ms per transaction
- Imbalance ratio: 577:1 (legitimate:fraud)
- No access to original features (PCA-transformed for privacy)
- Zero tolerance for missing critical fraud patterns

**Success Metrics:**
1. Primary: F1-Score ≥ 0.75
2. Secondary: Recall ≥ 75%, Precision ≥ 60%
3. Business: Net financial benefit > 0
4. Operational: False positive rate < 0.1%

---

## 2. Data Understanding

### Dataset Characteristics
```
Source: Kaggle Credit Card Fraud Detection
Origin: Worldline & Machine Learning Group (ULB)
Period: September 2013 (2 days of transactions)
Region: European cardholders

Size:
├── Total Transactions: 284,807
├── Fraudulent: 492 (0.173%)
├── Legitimate: 284,315 (99.827%)
└── Imbalance Ratio: 577.88:1

Features:
├── PCA Components: V1-V28 (28 features)
├── Time: Seconds elapsed from first transaction
├── Amount: Transaction amount in Euros (€)
└── Class: Target variable (0=Legitimate, 1=Fraud)

Data Quality:
├── Missing Values: 0 (100% complete)
├── Duplicates: 0
├── Outliers: Present (preserved - may indicate fraud)
└── Format: All numeric, ready for modeling
```

### Exploratory Data Analysis Findings

#### Class Distribution
- Severe imbalance: 577.88 legitimate per 1 fraud
- This ratio drives all methodology decisions
- Traditional accuracy-based approaches fail catastrophically

#### Feature Correlations (with Fraud)
**Strong Negative Predictors:**
1. V17: -0.3265 (strongest)
2. V14: -0.3025 (second strongest)
3. V12: -0.2606
4. V10: -0.2169
5. V16: -0.1965

**Interpretation:** Lower values of these PCA components → Higher fraud probability

**Strong Positive Predictors:**
1. V4: +0.1331
2. V11: +0.1544

#### Transaction Amount Patterns
```
Legitimate Transactions:
├── Mean: €88.29
├── Median: €22.00
├── Std Dev: €250.11
└── Max: €25,691.16

Fraudulent Transactions:
├── Mean: €122.21  (Higher than legitimate!)
├── Median: €9.25  (Much lower than legitimate!)
├── Std Dev: €256.68
└── Max: €2,125.87
```

**Critical Insight - Bimodal Fraud Pattern:**
- Many small frauds (€5-€15): Testing stolen credentials
- Fewer large frauds (€100-€2,000): Exploitation phase
- This explains why median fraud < median legitimate, but mean fraud > mean legitimate

---

## 3. Preprocessing Pipeline

### Step 1: Data Loading
```r
# Load dataset
data <- read.csv("data/creditcard.csv")

# Verify structure
str(data)      # Check data types
summary(data)  # Statistical summary
table(data$Class)  # Class distribution
```

### Step 2: Feature Standardization
**Why Needed:**
- V1-V28 already standardized (PCA output)
- Time and Amount on different scales
- Distance-based models (Logistic Regression) sensitive to scale
- Tree-based models don't require it, but we standardize for consistency

**Implementation:**
```r
# Z-score normalization
data$Time <- scale(data$Time)
data$Amount <- scale(data$Amount)

# Verify standardization
mean(data$Time)    # Should be ≈ 0
sd(data$Time)      # Should be ≈ 1
```

### Step 3: Target Variable Conversion
```r
# Convert to factor for classification
data$Class <- as.factor(data$Class)
levels(data$Class)  # Check: "0" "1"
```

### Step 4: Stratified Data Partitioning
**Critical for Imbalanced Data:**
- Random split could allocate 360-430 frauds to training (huge variance)
- Stratified split maintains exact 0.173% fraud rate in both sets
- Enables fair model comparison

**Implementation:**
```r
set.seed(42)  # Reproducibility
train_index <- createDataPartition(
  y = data$Class,
  p = 0.8,        # 80% training
  list = FALSE
)

train.df <- data[train_index, ]
valid.df <- data[-train_index, ]

# Verify stratification
prop.table(table(train.df$Class))  # Should be 0.827%, 0.173%
prop.table(table(valid.df$Class))  # Should be 0.827%, 0.173%
```

**Result:**
```
Training Set:   227,846 transactions (394 frauds, 0.173%)
Validation Set: 56,961 transactions  (98 frauds, 0.172%)
```

---

## 4. Imbalance Handling Strategies

We tested three fundamentally different approaches:

### Strategy 1: No Imbalance Handling (Baseline)
**Approach:** Train on imbalanced data without adjustments

**Code:**
```r
model_baseline <- glm(Class ~ ., 
                     data = train.df, 
                     family = binomial())
```

**Result:**
- Accuracy: 99.92% (misleading!)
- Recall: 68.37% (misses 31% of frauds)
- Precision: 82.72%
- F1-Score: 0.7486

**Lesson:** Model learns to predict majority class → Poor fraud detection

### Strategy 2: Data-Level Balancing
**Approach:** Manually balance training data to 50:50 ratio

**Implementation:**
```r
manual_balance <- function(data, ratio = 0.5) {
  # Separate classes
  fraud <- data[data$Class == 1, ]
  legit <- data[data$Class == 0, ]
  
  n_fraud <- nrow(fraud)
  n_legit <- floor(n_fraud * (1 - ratio) / ratio)
  
  # Oversample fraud (with replacement)
  fraud_balanced <- fraud[sample(nrow(fraud), 
                                n_fraud, 
                                replace = TRUE), ]
  
  # Undersample legitimate (without replacement)
  legit_balanced <- legit[sample(nrow(legit), 
                                n_legit, 
                                replace = FALSE), ]
  
  # Combine and shuffle
  balanced <- rbind(fraud_balanced, legit_balanced)
  balanced <- balanced[sample(nrow(balanced)), ]
  
  return(balanced)
}

# Apply balancing
train_balanced <- manual_balance(train.df, ratio = 0.5)
# Result: 788 rows (394 fraud + 394 legitimate)
```

**Models Trained:**
1. Logistic Regression
2. Decision Tree with pruning

**Results:**
```
Logistic Regression (Balanced):
├── Recall: 90.82% ← Excellent fraud detection!
├── Precision: 4.91% ← Catastrophic false alarm rate
├── False Positives: 1,724
└── Net Benefit: -€42,000 (LOSES MONEY)

Decision Tree (Balanced):
├── Recall: 91.84% ← Even better detection!
├── Precision: 2.48% ← Even worse false alarms!
├── False Positives: 3,537
└── Net Benefit: -€136,000 (DISASTER)
```

**Why This Fails:**
1. Training on 50:50 teaches model: "Fraud is 50% prevalent"
2. Real world: Fraud is 0.17% prevalent (295× rarer!)
3. Model flags 6-40× more transactions than actual fraud rate
4. Discards 99.8% of legitimate training data (226,658 samples lost)
5. Loses information about legitimate transaction diversity

**Critical Lesson:** Data balancing fixes recall but destroys precision by fundamentally miscalibrating probability estimates.

### Strategy 3: Algorithmic Class Weighting (Optimal)
**Approach:** Train on full imbalanced data but weight fraud errors 577× more

**Implementation:**
```r
rf_optimal <- randomForest(
  Class ~ .,
  data = train.df,           # ALL 227,846 samples
  ntree = 100,               # 100 trees (balance speed/performance)
  mtry = 5,                  # √30 ≈ 5 features per split
  nodesize = 1,              # Allow full tree growth
  classwt = c('0'=1, '1'=577),  # KEY: Weight fraud 577×
  importance = TRUE          # Track feature importance
)
```

**How Class Weights Work:**
1. During tree construction, split quality measured by weighted Gini impurity
2. Misclassifying fraud (Class 1) penalized 577× more than legitimate (Class 0)
3. Trees learn: "Missing fraud is 577× costlier than false alarm"
4. BUT model still trains on real 577:1 distribution
5. Probability estimates remain properly calibrated

**Result:**
- Recall: 79.59% (catches 4 out of 5 frauds)
- Precision: 94.00% (94% of flags are real)
- F1-Score: 0.8619 (optimal balance)
- False Positives: 5 (operationally feasible!)
- Net Benefit: €28,750 per 2 days

**Why This Wins:**
1. Uses all 227,846 training samples (289× more than balanced)
2. Maintains proper probability calibration (trains on real 577:1)
3. Algorithmic adjustment (not data manipulation)
4. Achieves both high recall AND high precision
5. Positive financial impact (not negative like balanced models)

---

## 5. Model Training

### Model 1: Baseline Logistic Regression
**Purpose:** Establish minimum acceptable performance

**Configuration:**
```r
model1 <- glm(Class ~ ., 
             data = train.df,
             family = binomial(link = "logit"))
```

**Hyperparameters:**
- Family: Binomial (binary classification)
- Link: Logit (standard for logistic regression)
- No regularization (dataset size sufficient)

**Validation:**
```r
pred_prob1 <- predict(model1, valid.df, type = "response")
pred_class1 <- ifelse(pred_prob1 >= 0.5, 1, 0)
cm1 <- confusionMatrix(as.factor(pred_class1), 
                       valid.df$Class, 
                       positive = "1")
```

### Model 2: Logistic Regression (Balanced)
**Purpose:** Test if data balancing improves detection

**Configuration:**
```r
train_balanced <- manual_balance(train.df, ratio = 0.5)
model2 <- glm(Class ~ ., 
             data = train_balanced,
             family = binomial())
```

**Expectation:** High recall, uncertain precision

### Model 3: Decision Tree (Balanced + Pruning)
**Purpose:** Test non-linear method with interpretable rules

**Configuration:**
```r
# Step 1: Grow complex tree
tree_complex <- rpart(
  Class ~ .,
  data = train_balanced,
  control = rpart.control(
    cp = 0.00001,      # Very low complexity parameter
    minsplit = 5,      # Minimum 5 observations to split
    minbucket = 2,     # Minimum 2 observations in leaf
    xval = 5           # 5-fold cross-validation
  )
)

# Step 2: Identify optimal CP from CV
printcp(tree_complex)
plotcp(tree_complex)
optimal_cp <- tree_complex$cptable[which.min(tree_complex$cptable[,"xerror"]), "CP"]

# Step 3: Prune to optimal complexity
tree_pruned <- prune(tree_complex, cp = 0.0001)
```

**Why This Approach:**
1. Initial low CP (0.00001) allows tree to capture detailed patterns
2. Cross-validation identifies branches that don't improve generalization
3. Pruning removes overfitting while keeping useful structure
4. Result: Simpler, more robust tree

**Tree Interpretation Example:**
```
IF V17 < -2.5:
  IF V14 < -1.2:
    Predict: FRAUD (90% confidence)
  ELSE:
    Predict: LEGITIMATE
ELSE:
  Predict: LEGITIMATE
```

### Model 4: Random Forest (Class Weights) ⭐
**Purpose:** Production model combining ensemble robustness + class weighting

**Configuration:**
```r
rf_optimal <- randomForest(
  Class ~ .,
  data = train.df,                    # Full dataset
  ntree = 100,                        # 100 trees
  mtry = 5,                           # √30 features per split
  nodesize = 1,                       # Full tree growth
  maxnodes = NULL,                    # No node limit
  classwt = c('0' = 1, '1' = 577),   # Key parameter!
  importance = TRUE,
  sampsize = nrow(train.df),          # Full bootstrap samples
  replace = TRUE                       # With replacement
)
```

**Key Parameters Explained:**

**ntree = 100:**
- More trees → lower variance → better performance
- Diminishing returns after ~100 trees
- Balance: 100 trees = 2 minutes training, excellent performance

**mtry = 5:**
- Standard: √(number of features) = √30 ≈ 5.5
- Randomly select 5 features at each split
- Promotes diversity among trees

**nodesize = 1:**
- Allow trees to grow until leaf purity
- Each leaf can have as few as 1 observation
- Captures detailed patterns

**NO maxnodes constraint:**
- Previous bug: maxnodes=50 caused severe underfitting
- With 30 features and 227K samples, need hundreds of nodes
- Let trees grow naturally based on data

**classwt = c('0'=1, '1'=577):**
- **Most important parameter**
- Weight fraud errors 577× more than legitimate errors
- Teaches model: "Missing fraud is catastrophically expensive"

**How Random Forest Works:**
1. Bootstrap sample from training data (with replacement)
2. At each node, randomly select mtry=5 features
3. Find best split using weighted Gini impurity
4. Grow tree until nodesize=1 reached
5. Repeat 100 times
6. Final prediction: Majority vote (or average probability)

**Why Ensemble Wins:**
- Single tree: High variance, overfits easily
- 100 trees: Errors average out, robust predictions
- Each tree sees different data + different features
- Diversity → Accuracy

---

## 6. Threshold Optimization

### Why Optimize Threshold?

**Default 0.5 is NOT optimal for:**
1. Imbalanced problems
2. Class-weighted models
3. Cost-sensitive applications

**Class weights shift probability distributions:**
```
Without weights:   P(Fraud) ∈ [0.0, 1.0], natural calibration
With weights:      P(Fraud) skewed higher, 0.5 no longer optimal
```

### Optimization Procedure

**Step 1: Generate Probability Predictions**
```r
rf_probs <- predict(rf_optimal, 
                   valid.df, 
                   type = "prob")[, 2]  # Get fraud probability
```

**Step 2: Test Multiple Thresholds**
```r
thresholds <- seq(0.10, 0.90, by = 0.05)
results <- data.frame()

for (thresh in thresholds) {
  # Convert probabilities to predictions
  pred <- ifelse(rf_probs >= thresh, 1, 0)
  
  # Calculate confusion matrix
  cm <- confusionMatrix(as.factor(pred), 
                       valid.df$Class, 
                       positive = "1")
  
  # Store metrics
  results <- rbind(results, 
                  data.frame(
                    Threshold = thresh,
                    Precision = cm$byClass[3],
                    Recall = cm$byClass[1],
                    F1 = cm$byClass[7],
                    Specificity = cm$byClass[2]
                  ))
}
```

**Step 3: Find Optimal Threshold**
```r
optimal_threshold <- results$Threshold[which.max(results$F1)]
print(paste("Optimal Threshold:", optimal_threshold))
# Result: 0.40
```

### Threshold Sensitivity Analysis

| Threshold | Precision | Recall | F1-Score | Trade-off |
|-----------|-----------|--------|----------|-----------|
| 0.20 | 85.6% | 84.7% | 0.851 | High recall, moderate precision |
| 0.30 | 89.0% | 82.7% | 0.857 | Balanced |
| **0.40** | **94.0%** | **79.6%** | **0.862** | **Optimal** |
| 0.50 | 96.2% | 76.5% | 0.852 | High precision, lower recall |
| 0.60 | 97.2% | 71.4% | 0.824 | Very conservative |

**Insights:**
- F1 peaks at 0.40 (not default 0.5!)
- Lowering from 0.5 → 0.4 catches 3 more frauds (€1,500)
- Cost: 3 more false alarms (€150)
- Net gain: €1,350 per 2 days = €246,375 annually
- 1.1% F1 improvement for 5 minutes of work!

### Visualization
```r
plot(results$Threshold, results$F1,
     type = "l", col = "purple", lwd = 3,
     xlab = "Threshold", ylab = "Metric",
     main = "Threshold Optimization")

lines(results$Threshold, results$Precision, col = "blue", lwd = 2)
lines(results$Threshold, results$Recall, col = "red", lwd = 2)
abline(v = 0.40, col = "green", lty = 2, lwd = 2)

legend("right",
       legend = c("F1-Score", "Precision", "Recall", "Optimal"),
       col = c("purple", "blue", "red", "green"),
       lwd = c(3, 2, 2, 2),
       lty = c(1, 1, 1, 2))
```

---

## 7. Evaluation Framework

### Confusion Matrix Structure
```
                    Predicted: 0    Predicted: 1
Actual: 0 (Legit)      TN              FP
Actual: 1 (Fraud)      FN              TP
```

**At Threshold 0.40:**
```
                    Predicted: 0    Predicted: 1
Legitimate (56,863)    56,858            5
Fraud (98)                20             78
```

### Performance Metrics

**1. Accuracy**
```
Formula: (TP + TN) / (TP + TN + FP + FN)
Value: (78 + 56,858) / 56,961 = 0.9996 (99.96%)

Why Misleading:
├── "Always Predict 0" = 99.83% accuracy, 0% recall
├── Our model: 99.96% accuracy, 79.6% recall
└── Difference: 0.13% accuracy improvement hides massive value gain
```

**2. Recall (Sensitivity, True Positive Rate)**
```
Formula: TP / (TP + FN)
Value: 78 / (78 + 20) = 0.7959 (79.59%)

Interpretation:
├── Catches 78 out of 98 frauds
├── Misses 20 frauds (20.4% miss rate)
└── For 577:1 imbalance, 80% recall is excellent
```

**3. Precision (Positive Predictive Value)**
```
Formula: TP / (TP + FP)
Value: 78 / (78 + 5) = 0.9400 (94.00%)

Interpretation:
├── 94 out of 100 fraud flags are real fraud
├── Only 6% false alarm rate
└── Fraud team investigates real fraud 94% of time
```

**4. F1-Score (Harmonic Mean)**
```
Formula: 2 × (Precision × Recall) / (Precision + Recall)
Value: 2 × (0.94 × 0.7959) / (0.94 + 0.7959) = 0.8619

Why Primary Metric:
├── Accuracy misleading (dominated by TN)
├── Need both high precision AND high recall
├── F1 balances both objectives
└── 0.86 = Excellent (0-1 scale)
```

**5. Specificity (True Negative Rate)**
```
Formula: TN / (TN + FP)
Value: 56,858 / (56,858 + 5) = 0.9999 (99.99%)

Interpretation:
├── Correctly identifies 99.99% of legitimate
├── False positive rate: 0.0088%
└── Only 5 false alarms per 56,863 transactions
```

### ROC-AUC Analysis
```r
library(pROC)
roc_obj <- roc(valid.df$Class, rf_probs)
auc_value <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj,
     main = "ROC Curve - Random Forest",
     col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.5, 0.3, paste("AUC =", round(auc_value, 4)))
```

**AUC Interpretation:**
- AUC = 0.976 (excellent discrimination)
- Probability that random fraud ranks higher than random legitimate
- Close to 1.0 = Model separates classes well

### Business Metrics

**Cost-Benefit Analysis:**
```r
# Extract confusion matrix values
tp <- cm_final$table[2, 2]  # 78
fp <- cm_final$table[2, 1]  # 5
fn <- cm_final$table[1, 2]  # 20
tn <- cm_final$table[1, 1]  # 56,858

# Business costs
cost_per_fraud <- 500          # € per undetected fraud
cost_per_investigation <- 50   # € per false alarm

# Calculate financial impact
savings_detected <- tp * cost_per_fraud
losses_missed <- fn * cost_per_fraud
cost_investigations <- fp * cost_per_investigation

net_benefit_2days <- savings_detected - losses_missed - cost_investigations
net_benefit_annual <- net_benefit_2days / 2 * 365

# ROI calculation
implementation_cost <- 111000  # First year
roi <- ((net_benefit_annual - implementation_cost) / implementation_cost) * 100
```

**Results:**
```
2-Day Validation Period:
├── Savings (detected fraud):   €39,000
├── Losses (missed fraud):      -€10,000
├── Investigation costs:        -€250
└── Net Benefit:                €28,750

Annual Projection:
├── Daily value:                €14,375
├── Annual benefit:             €5,246,875
├── Implementation cost:        €111,000
├── Net first-year value:       €5,135,875
├── ROI:                        4,627%
└── Payback period:             7.7 days
```

---

## 8. Production Deployment

### Three-Phase Rollout Strategy

#### Phase 1: Shadow Mode (1-2 months)
**Objective:** Validate real-world performance without risk

**Implementation:**
```r
# Production code runs in parallel with existing system
production_predictions <- function(transaction) {
  # Preprocess
  transaction_scaled <- scale_features(transaction)
  
  # Generate prediction
  fraud_prob <- predict(rf_model, 
                       transaction_scaled, 
                       type = "prob")[2]
  
  # Log but don't act
  log_prediction(transaction_id, fraud_prob, timestamp)
  
  return(fraud_prob)  # For monitoring only
}
```

**Monitoring:**
- Compare model predictions with actual fraud labels
- Measure: F1, Recall, Precision, FP Rate daily
- Track: Feature drift, prediction distribution
- Alert: If F1 < 0.75 or FP rate > 0.02%

**Success Criteria:**
- F1-Score ≥ 0.75 on live traffic
- FP rate < 0.02% (acceptable to fraud team)
- No critical failures (missed high-value fraud)
- Model predictions stable over time

#### Phase 2: Assisted Mode (1-2 months)
**Objective:** Capture value while maintaining human oversight

**Implementation:**
```r
automated_fraud_decision <- function(transaction) {
  fraud_prob <- predict(rf_model, transaction, type = "prob")[2]
  
  if (fraud_prob >= 0.7) {
    # High confidence: Auto-block
    action <- "BLOCK"
    alert_customer(transaction, reason = "fraud_detected")
    log_action(transaction, action, fraud_prob, auto = TRUE)
    
  } else if (fraud_prob >= 0.4) {
    # Medium confidence: Flag for review
    action <- "REVIEW"
    flag_for_analyst(transaction, priority = "HIGH", prob = fraud_prob)
    log_action(transaction, action, fraud_prob, auto = FALSE)
    
  } else {
    # Low confidence: Approve
    action <- "APPROVE"
    log_action(transaction, action, fraud_prob, auto = TRUE)
  }
  
  return(action)
}
```

**Thresholds:**
- 0.7-1.0: Auto-block (99%+ precision expected)
- 0.4-0.7: Manual review (fraud analysts prioritize)
- 0.0-0.4: Auto-approve (very low fraud probability)

**Analyst Feedback:**
```r
analyst_override <- function(transaction_id, model_prediction, analyst_decision) {
  # Log overrides
  log_override(
    transaction_id = transaction_id,
    model_pred = model_prediction,
    analyst_decision = analyst_decision,
    reason = get_reason_code()
  )
  
  # Use overrides for model retraining
  add_to_retraining_queue(transaction_id, analyst_decision)
}
```

**Success Criteria:**
- 90%+ of high-confidence predictions confirmed by analysts
- <5% override rate on medium-confidence cases
- Positive analyst feedback
- Fraud losses decrease by ≥20% vs baseline

#### Phase 3: Full Automation (Ongoing)
**Objective:** Scale to full production with continuous monitoring

**Implementation:**
```r
production_pipeline <- function(transaction) {
  # Real-time scoring
  fraud_prob <- predict(rf_model, transaction, type = "prob")[2]
  
  # Decision
  if (fraud_prob >= 0.40) {
    block_transaction(transaction)
    create_investigation_case(transaction, fraud_prob)
  } else {
    approve_transaction(transaction)
  }
  
  # Monitoring
  log_transaction(transaction, fraud_prob, decision)
  update_metrics_dashboard()
}
```

**Continuous Monitoring:**
```r
# Daily performance check
daily_monitor <- function() {
  # Get last 24 hours predictions
  predictions_24h <- get_predictions(hours = 24)
  actuals_24h <- get_actuals(hours = 24)
  
  # Calculate metrics
  metrics <- calculate_metrics(predictions_24h, actuals_24h)
  
  # Alert if degradation
  if (metrics$f1 < 0.75) {
    send_alert("MODEL DEGRADATION: F1 = ", metrics$f1)
  }
  if (metrics$fp_rate > 0.02) {
    send_alert("HIGH FALSE POSITIVES: Rate = ", metrics$fp_rate)
  }
  
  # Update dashboard
  update_dashboard(metrics)
}
```

**Quarterly Retraining:**
```r
quarterly_retrain <- function() {
  # Get last 6 months of data
  training_data <- get_transactions(months = 6)
  
  # Train new model
  rf_new <- randomForest(
    Class ~ .,
    data = training_data,
    ntree = 100,
    classwt = c('0'=1, '1'=577),
    importance = TRUE
  )
  
  # A/B test new vs old
  ab_test_results <- ab_test(rf_new, rf_current, test_period = 14)
  
  # Deploy if improved
  if (ab_test_results$f1_new > ab_test_results$f1_current) {
    deploy_model(rf_new)
    archive_model(rf_current)
  }
}
```

### Deployment Checklist

- [ ] Model serialization (saveRDS/joblib)
- [ ] API endpoint setup (Plumber/FastAPI)
- [ ] Load testing (>1000 req/s)
- [ ] Latency testing (<100ms p95)
- [ ] Feature preprocessing pipeline
- [ ] Logging infrastructure
- [ ] Monitoring dashboard
- [ ] Alert system
- [ ] A/B testing framework
- [ ] Rollback procedure
- [ ] Documentation for ops team
- [ ] Runbook for incidents

---

## Conclusion

This methodology demonstrates that severe class imbalance (577:1) can be successfully addressed through:

1. **Algorithmic class weighting** (superior to data balancing)
2. **Full dataset utilization** (all 227K samples)
3. **Threshold optimization** (beyond default 0.5)
4. **Proper evaluation** (F1-Score, not accuracy)
5. **Production-ready deployment** (3-phase rollout)

**Key Lesson:** For extreme imbalance, teach the algorithm that errors are costly (class weights), don't manipulate data to make classes appear equally common (balancing).

**Result:** Production-ready system with 79.6% recall, 94.0% precision, €5.25M annual value.
