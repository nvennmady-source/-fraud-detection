# 🏦 Credit Card Fraud Detection with Genetic Algorithm

This project applies **Genetic Algorithm (GA)** for **feature selection** and evaluates multiple machine learning models for **credit card fraud detection**.

## 📊 Dataset
The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.  
It contains transactions made by European cardholders in September 2013, with **492 frauds out of 284,807 transactions**.

⚠️ The dataset is too large to upload to GitHub. Please download it from Kaggle and place it in the project root as:creditcard.csv



## ⚙️ Installation
Clone this repo and install the required dependencies:
```bash
git clone https://github.com/nvennmady-source/fraud-detection-ga.git
cd fraud-detection-ga
pip install -r requirements.txt


## Methodology

Genetic Algorithm (GA) is used to select the most relevant features.

The following models are trained and evaluated:

Random Forest

Decision Tree

Artificial Neural Network (MLPClassifier)

Naive Bayes

Logistic Regression

Performance metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

 ## 📈 Results

Example metrics from different feature vectors:

Random Forest achieved the highest accuracy and recall.

ANN provided competitive performance with balanced F1-score.
