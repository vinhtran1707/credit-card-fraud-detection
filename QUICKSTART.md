# Quick Start Guide

This guide will get you running the fraud detection model in under 10 minutes.

## Prerequisites

- **R:** Version 4.0 or higher
- **RStudio:** Latest version (recommended)
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** ~500MB free space

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

## Step 2: Install Dependencies

Open R or RStudio and run:

```r
# Install all required packages
install.packages(c(
  "tidyverse",      # Data manipulation (dplyr, ggplot2, etc.)
  "caret",          # Machine learning workflow
  "randomForest",   # Random Forest algorithm
  "rpart",          # Decision trees
  "rpart.plot",     # Tree visualization
  "pROC",           # ROC curve analysis
  "ROSE",           # Sampling techniques
  "MLmetrics",      # Performance metrics
  "corrplot",       # Correlation plots
  "gridExtra",      # Multiple plots
  "scales"          # Visualization scales
))
```

**Installation time:** ~5-10 minutes depending on your connection

## Step 3: Download Dataset

### Option A: Kaggle Website (Easiest)
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" (requires Kaggle account)
3. Extract `creditcard.csv` to `data/` folder

### Option B: Kaggle API (Faster)
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud

# Extract to data folder
unzip creditcardfraud.zip -d data/
```

## Step 4: Run Complete Analysis

```r
# Set working directory
setwd("path/to/credit-card-fraud-detection")

# Run full pipeline
source("scripts/FINAL_VERSION_CCCODES.R")
```

**Expected output:**
- Console logs showing progress
- Visualizations displayed in Plots pane
- Results saved to `outputs/` folder
- Runtime: ~5-10 minutes

## Step 5: View Results

### Check Console Output
```r
# You should see:
# ✓ Data loaded: 284,807 transactions
# ✓ Training set: 227,846 (394 frauds)
# ✓ Validation set: 56,961 (98 frauds)
# ✓ Model trained: Random Forest (100 trees)
# ✓ Optimal threshold: 0.40
# ✓ F1-Score: 0.8619
# ✓ Net Benefit: €28,750
```

### Check Generated Files
```
outputs/
├── confusion_matrix_rf.txt
├── model_performance.csv
├── threshold_optimization.csv
└── predictions.csv
```

### View Visualizations
Check the `images/` folder for:
- Class distribution
- Feature correlations
- ROC curves
- Threshold optimization plot
- Variable importance
- Decision tree visualization

## Common Commands

### Load Pre-trained Model
```r
# Load saved model
rf_model <- readRDS("models/rf_optimal.rds")

# Make predictions
new_transaction <- data.frame(
  Time = 0.5,
  V1 = -1.35,
  V2 = -0.07,
  # ... (all 30 features)
  Amount = 149.62
)

fraud_prob <- predict(rf_model, new_transaction, type = "prob")[2]
print(paste("Fraud Probability:", round(fraud_prob, 4)))
```

### Generate Specific Visualization
```r
# ROC Curve
source("scripts/visualizations/plot_roc.R")

# Threshold Optimization
source("scripts/visualizations/plot_threshold.R")

# Confusion Matrix
source("scripts/visualizations/plot_confusion.R")
```

### Run Individual Models
```r
# Baseline Logistic Regression
source("scripts/models/01_logistic_baseline.R")

# Balanced Logistic Regression
source("scripts/models/02_logistic_balanced.R")

# Decision Tree
source("scripts/models/03_decision_tree.R")

# Random Forest (Optimal)
source("scripts/models/04_random_forest.R")
```

## Troubleshooting

### Issue: Package installation fails
```r
# Try installing from CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com/"))
install.packages("package_name")
```

### Issue: Memory error
```r
# Reduce Random Forest size
rf_model <- randomForest(..., ntree = 50)  # Instead of 100
```

### Issue: "File not found" error
```r
# Check working directory
getwd()

# Set correct directory
setwd("path/to/project")

# Verify data file exists
file.exists("data/creditcard.csv")  # Should return TRUE
```

### Issue: Slow performance
```r
# Use parallel processing
library(doParallel)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Your code here

stopCluster(cl)
```

## Next Steps

### 1. Explore the Code
- Read `scripts/FINAL_VERSION_CCCODES.R` to understand pipeline
- Check `METHODOLOGY.md` for detailed explanations
- Review `docs/FINAL_Report.docx` for comprehensive analysis

### 2. Experiment
```r
# Try different thresholds
threshold <- 0.30  # More sensitive
threshold <- 0.50  # More conservative

# Try different class weights
classwt = c('0'=1, '1'=400)  # Less aggressive
classwt = c('0'=1, '1'=700)  # More aggressive

# Try different ensemble sizes
ntree = 50   # Faster
ntree = 200  # More accurate (slower)
```

### 3. Customize
- Modify cost assumptions (€500/fraud, €50/false alarm)
- Add new visualizations
- Implement additional models (XGBoost, Neural Networks)
- Export predictions for business intelligence tools

## Performance Benchmarks

**On typical laptop (Intel i5, 8GB RAM):**
- Data loading: 5-10 seconds
- Preprocessing: 2-5 seconds
- Model training (RF 100 trees): 2-3 minutes
- Threshold optimization: 30-60 seconds
- Total runtime: 5-8 minutes

**Expected Results:**
- F1-Score: 0.86-0.87
- Recall: 79-80%
- Precision: 93-95%
- False Positives: 5-7 (per 56,961 validation transactions)

## Getting Help

### Documentation
- **README.md:** Project overview
- **METHODOLOGY.md:** Detailed technical documentation
- **docs/FINAL_Report.docx:** Full academic report

### Issues
- Check existing issues: https://github.com/yourusername/credit-card-fraud-detection/issues
- Open new issue with:
  - Error message
  - System information (R version, OS)
  - Steps to reproduce

### Contact
- Email: vtran13@tulane.edu
- LinkedIn: [Your Profile]
- GitHub: @yourusername

---

## Quick Reference Card

```r
# ESSENTIAL COMMANDS

# 1. Setup
setwd("path/to/project")
source("scripts/00_load_libraries.R")

# 2. Load Data
data <- read.csv("data/creditcard.csv")

# 3. Train Model
source("scripts/FINAL_VERSION_CCCODES.R")

# 4. Check Results
print(paste("F1-Score:", final_f1))
print(paste("Net Benefit:", net_benefit))

# 5. Make Prediction
new_pred <- predict(rf_model, new_data, type="prob")[,2]
is_fraud <- new_pred >= 0.40
```

---

**You're ready to go!** 🚀

If everything works, you should see model performance metrics and visualizations within 10 minutes.
