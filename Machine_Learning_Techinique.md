# 🧠 Machine Learning & Data Science Complete Reference Guide

A concise, revision-ready, exam-oriented reference covering the end‑to‑end ML workflow: from data cleaning to deployment.

---

# 📋 Table of Contents

1. Data Cleaning
2. Data Preprocessing Pipeline
3. Feature Engineering
4. Model Training Techniques
5. Model Selection & Validation
6. Model Evaluation Metrics
7. Hyperparameter Tuning
8. Model Deployment
9. Visualization Techniques

---

# 🧹 1. Data Cleaning

### Core Cleaning Steps

* Handle missing values: mean/median/mode, KNN, interpolation, model‑based, missingness flag
* Remove duplicates: exact, fuzzy, keep first/last
* Outlier detection: IQR, Z‑score, Isolation Forest → cap/winsorize/remove
* Fix data types: numeric conversion, datetime parsing, boolean fixing
* Datetime cleanup: timezone, components, invalid date handling
* Text normalization: lowercasing, whitespace, special chars, spelling correction
* Standardize categories: unify inconsistent labels
* Clean column names: snake_case, remove spaces
* Validate values: impossible values, logic checks

### Advanced Cleaning

* Encoding issues: UTF‑8 conversion, mojibake fixes
* Placeholder values: replace -999, "NA", "Unknown"
* Unit standardization: km→miles, kg→lbs, currency conversion
* Cross-field validation: start_date < end_date
* Schema validation: type checks, required fields
* Rare category merging
* Handle skewness: log, Box‑Cox, Yeo‑Johnson

---

# ⚙️ 2. Data Preprocessing Pipeline

### Workflow

**1. Understanding**: Explore shape, dtypes, stats, quality issues

**2. Core Cleaning**: duplicates → placeholders → types → text → categories → units

**3. Quality Control**: mixed types, outliers, missing value strategy, consistency checks

**4. Feature Preparation**:

* Encoding: one‑hot, label, target, frequency, hash
* Scaling: Standard, MinMax, Robust
* Class imbalance handling: SMOTE, undersampling, class weights
* Dimensionality reduction/selection

**5. Finalization**:

* Correct data splits (stratified/time‑based)
* Build reusable preprocessing pipelines
* Save scalers/encoders
* Validate transformations
* Monitor drift

---

# 🧩 3. Feature Engineering

### Feature Creation

* Interaction: A×B, A/B, A−B
* Polynomial features
* Aggregations: groupby means, counts, windows
* Combined categorical features

### Numerical Transformations

* Scaling: StandardScaler, MinMaxScaler, RobustScaler
* Log / sqrt / Box‑Cox / Yeo‑Johnson
* Binning: equal‑width, equal‑frequency
* Quantile transforms

### Categorical Encoding

* Label, One‑Hot, Target, Frequency
* Binary, Hashing, Leave‑One‑Out

### Time Series Features

* Date parts, cyclical encoding (sin/cos)
* Lags, rolling stats, rolling window features
* Time differences, holiday flags

### Text Features

* BoW, TF‑IDF, n‑grams
* Embeddings: Word2Vec, FastText, BERT
* Text stats: length, counts
* Sentiment features

### Geospatial Features

* Haversine distance
* Region clustering
* Density + proximity features

### Dimensionality Reduction

* PCA, SVD, LDA
* Autoencoders
* t‑SNE/UMAP (visualization)

### Feature Selection

* Filter: correlation, variance threshold
* Wrapper: RFE, forward/backward selection
* Embedded: Lasso, Tree‑based importances

---

# 🤖 4. Model Training Techniques

### Supervised

* Regression: Linear, Ridge, Lasso, ElasticNet, Tree‑based, Boosting
* Classification: Logistic, SVM, Decision Trees, Random Forest, XGBoost/LightGBM/CatBoost

### Unsupervised

* Clustering: K‑Means, DBSCAN, GMM
* Dimensionality reduction: PCA, ICA, UMAP

### Semi‑Supervised

* Pseudo‑labeling

### Reinforcement Learning

* Q‑Learning, DQN, PPO, Actor‑Critic

### Optimization Algorithms

* SGD, Momentum, Nesterov, Adam/AdamW, RMSProp, Adagrad, L‑BFGS

### Regularization

* L1, L2, Elastic Net, Dropout, Early stopping, Weight decay

### Deep Learning

* LR schedules: step, cosine, warmup
* Initialization: Xavier, He
* Batch/Layer Normalization
* Transfer Learning, Fine‑tuning
* GAN/Diffusion basics

### Advanced Training

* Curriculum learning
* Multi‑task learning
* Distillation
* Online/Incremental training
* Distributed training (DDP, Horovod, DeepSpeed)

---

# 🔍 5. Model Selection & Validation

### Cross-Validation

* Train/test split
* K‑Fold, Stratified K‑Fold
* LOOCV, Time series split (rolling/expanding)
* Nested CV for unbiased tuning

### Hyperparameter Methods

* Grid Search, Random Search
* Bayesian Optimization, TPE
* Hyperband, ASHA, BOHB

### Model Selection Criteria

**Classification**: Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC

**Regression**: MAE, RMSE, R²

**Information Criteria**: AIC, BIC

### Ensemble Methods

* Bagging, Boosting, Stacking, Blending

### Statistical Tests

* Chi‑Square, ANOVA, Kolmogorov‑Smirnov

### Robustness Checks

* Learning curves
* Validation curves
* Residual analysis

---

# 📊 6. Model Evaluation Metrics

### Classification

* Accuracy, Precision, Recall, F1
* ROC‑AUC, PR‑AUC
* MCC, Cohen’s Kappa
* Confusion Matrix

### Regression

* MAE, MSE, RMSE
* MAPE, SMAPE
* R², Adjusted R²

### Clustering

* Silhouette score
* Davies‑Bouldin index
* Calinski‑Harabasz index

### Time Series

* MAE, RMSE, MAPE, SMAPE
* Theil’s U, MASE

### NLP/CV

* BLEU, ROUGE, METEOR, BERTScore
* FID, IS (GAN metrics)

### Interpretability

* SHAP, LIME, Permutation importance
* PDP, ICE plots

---

# 🔧 7. Hyperparameter Tuning

### Core Methods

* Grid Search
* Random Search

### Advanced

* Bayesian Optimization
* TPE
* SMBO

### Efficient Methods

* Hyperband
* ASHA
* BOHB

### Deep Learning

* Keras Tuner
* Population‑based training
* LR Finder

---

# 🚀 8. Model Deployment

### Deployment Approaches

* REST APIs (Flask/FastAPI)
* Batch predictions
* Scheduled pipelines
* Dockerized services
* Kubernetes for scaling
* Cloud deployment (AWS, Azure, GCP)

### Production Needs

* Logging & monitoring
* Data drift detection
* Model versioning (MLflow)
* CI/CD automation

---

# 📈 9. Visualization Techniques

### Core Tools

* Matplotlib, Seaborn, Plotly
* Dash/Streamlit for web dashboards

### Plots

* Distribution: hist, kde, boxplot
* Relationships: scatter, heatmap, pairplot
* Time series: line, rolling windows
* Model insights: SHAP plots, feature importance

---

# ✔️ Summary

This reference is intentionally concise, exam‑focused, and optimized for fast revision. It can be used for interviews, deep study, and end‑to‑end ML project execution.

---

**End of Document**
