# Data Directory

## Dataset Information

### Source
- **Dataset Name:** Credit Card Fraud Detection
- **Provider:** Machine Learning Group (Université Libre de Bruxelles)
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### How to Download

#### Option 1: Manual Download
1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click "Download" button (requires Kaggle account)
3. Extract `creditcard.csv` to this directory

#### Option 2: Kaggle API (Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials
# Download kaggle.json from Kaggle Account Settings
# Move to ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud

# Extract
unzip creditcardfraud.zip -d data/
rm creditcardfraud.zip
```

### File Structure
```
data/
├── README.md                 # This file
└── creditcard.csv           # Main dataset (download separately)
```

### Dataset Characteristics

**Size Information:**
- File Size: ~150 MB
- Rows: 284,807 transactions
- Columns: 31 (28 PCA features + Time + Amount + Class)

**Features:**

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| Time | Numeric | Seconds elapsed since first transaction | 0 - 172,792 |
| V1-V28 | Numeric | PCA-transformed features (confidential) | Standardized |
| Amount | Numeric | Transaction amount in € | 0 - 25,691.16 |
| Class | Binary | Target variable (0=Legitimate, 1=Fraud) | 0, 1 |

**Class Distribution:**
- Legitimate (Class 0): 284,315 (99.827%)
- Fraudulent (Class 1): 492 (0.173%)
- Imbalance Ratio: 577.88:1

**Data Quality:**
- Missing Values: 0
- Duplicates: 0
- Outliers: Present (preserved as potential fraud indicators)
- Preprocessing: PCA already applied (V1-V28)

### Important Notes

⚠️ **Privacy:** Original features masked via PCA transformation for confidentiality

⚠️ **Large File:** creditcard.csv (~150MB) is not included in repository due to size

⚠️ **License:** Dataset available for research and educational purposes

### Verification

After downloading, verify your data:

```r
# Load and check
data <- read.csv("data/creditcard.csv")

# Expected output:
nrow(data)  # Should be 284,807
ncol(data)  # Should be 31
table(data$Class)  # Should show 284,315 (class 0) and 492 (class 1)
```

### Citation

If you use this dataset, please cite:

```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. 
Calibrating Probability with Undersampling for Unbalanced Classification. 
In Symposium on Computational Intelligence and Data Mining (CIDM), 
IEEE, 2015
```

### Need Help?

If you encounter issues downloading:
1. Ensure you have a Kaggle account
2. Accept the dataset terms and conditions
3. Check your Kaggle API credentials
4. See main README for troubleshooting

### Sample Data Structure

```csv
Time,V1,V2,V3,...,V28,Amount,Class
0,-1.3598071336738,-0.0727811733098497,2.53634673796914,...,0.133558376740387,149.62,0
0,1.19185711131486,0.26615071205963,0.16648011335321,...,-0.0089831888579,-2.261857087,-0.0089831888579,2.69,0
1,-1.35835406159823,-1.34016307473609,1.77320934263119,...,-0.0597518400105609,378.66,0
...
```

---

**Note:** The actual data file is not included in this repository. Please download it separately following the instructions above.
