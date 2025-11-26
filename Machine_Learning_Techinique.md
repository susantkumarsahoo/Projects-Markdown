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

# 🎯 Complete Machine Learning & Deep Learning Reference Guide

**A comprehensive guide covering the entire ML/DL pipeline - optimized for revision and exam preparation**

---

## 📋 Table of Contents

1. [Data Cleaning](#1-data-cleaning)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Selection & Cross-Validation](#4-model-selection--cross-validation)
5. [Training Techniques](#5-training-techniques)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Model Evaluation](#7-model-evaluation)
8. [Model Deployment](#8-model-deployment)
9. [Visualization Techniques](#9-visualization-techniques)
10. [Best Practices & Common Pitfalls](#10-best-practices--common-pitfalls)

---

## 1. Data Cleaning

### Core Topics
- **Missing Values** - Identify and handle NaN/null entries
- **Duplicates** - Remove exact/partial duplicates
- **Outliers** - Detect extreme values using IQR, Z-score
- **Wrong Data Types** - Convert to appropriate types
- **Datetime Cleaning** - Parse and standardize dates
- **Inconsistent Data Entry** - Standardize categorical values
- **Column Name Cleaning** - Remove spaces, special chars
- **Mixed Data Types** - Handle columns with multiple types
- **Text Normalization** - Lowercase, remove whitespace
- **Invalid Values** - Identify impossible/illogical values
- **Data Reduction** - Remove unnecessary rows/columns
- **Noise Removal** - Smooth or filter noisy data

### Advanced Topics
- **Encoding Issues** - Handle UTF-8, ASCII problems
- **Schema Validation** - Ensure data matches expected format
- **Cross-field Validation** - Check logical consistency
- **Unit Standardization** - Normalize measurement units
- **Placeholder Values** - Handle sentinel values (-999, "N/A")
- **Feature Consistency** - Ensure features align across datasets
- **Categorical Merging** - Combine similar categories
- **Outlier Capping** - Apply Winsorization or clipping

---

## 2. Data Preprocessing

### Step-by-Step Workflow
1. **Understand the Data** - EDA, distributions, correlations
2. **Define Goals & Constraints** - Problem type, metrics
3. **Initial Quality Checks** - Check shape, dtypes, nulls
4. **Remove Duplicates** - Drop exact duplicate rows
5. **Handle Placeholders** - Replace sentinel values
6. **Fix Data Types** - Convert to correct types
7. **Parse Datetime** - Extract date components
8. **Standardize Column Names** - Clean and consistent naming
9. **Text Cleaning** - Normalize text data
10. **Consistent Categories** - Standardize categorical values
11. **Unit Standardization** - Convert to common units
12. **Handle Mixed Types** - Resolve type conflicts
13. **Outlier Treatment** - Cap, remove, or transform
14. **Missing Value Strategy** - Impute or remove
15. **Cross-field Validation** - Check logical relationships
16. **Encode Categories** - Convert to numeric
17. **Feature Scaling** - Normalize/standardize
18. **Feature Engineering** - Create new features
19. **Dimensionality Reduction** - PCA, feature selection
20. **Handle Class Imbalance** - SMOTE, undersampling, class weights
21. **Split Data** - Train/validation/test sets
22. **Pipeline Creation** - Automate preprocessing steps
23. **Validate Transformations** - Check consistency
24. **Quality Assertions** - Test data integrity
25. **Document Steps** - Record all transformations
26. **Save Artifacts** - Save scalers, encoders, models
27. **Monitor Drift** - Track data distribution changes
28. **Package for Modeling** - Prepare final dataset

---

## 3. Feature Engineering

### A. Feature Creation
- **Domain Knowledge Features** - Business logic-based features
- **Interaction Features** - A × B, A / B, A + B
- **Polynomial Features** - x², x³, cross-terms
- **Aggregate Features** - sum, mean, max, min, count, std
- **Combination Features** - Concatenate categorical variables
- **Temporal Aggregates** - Daily, weekly, monthly stats

### B. Numerical Transformations

#### Scaling/Normalization
- **MinMaxScaler** - Scale to [0,1] range
- **StandardScaler** - Mean=0, Std=1 (Z-score normalization)
- **RobustScaler** - Uses median/IQR (handles outliers)
- **MaxAbsScaler** - Scale by maximum absolute value

#### Power Transforms
- **Log Transformation** - log(x) for right-skewed data
- **Square Root / Cube Root** - Moderate skewness
- **Box-Cox** - Automatic power transformation (positive data)
- **Yeo-Johnson** - Works with zero/negative values

#### Discretization
- **Binning** - Convert continuous to categorical
- **Rank Transformation** - Convert values to ranks
- **Quantile Transformation** - Uniform/normal distribution mapping

### C. Categorical Encoding
- **Label Encoding** - Integer codes (0, 1, 2...)
- **One-Hot Encoding** - Binary columns for each category
- **Target/Mean Encoding** - Replace with target mean
- **Frequency Encoding** - Replace with occurrence frequency
- **Binary Encoding** - Binary representation
- **Ordinal Encoding** - For ordered categories
- **Hash Encoding** - For high-cardinality features
- **Leave-One-Out Encoding** - Avoid target leakage

### D. Handling Missing Values
- **Deletion** - Drop rows/columns with missing data
- **Mean/Median/Mode Imputation** - Fill with statistics
- **Forward/Backward Fill** - Use adjacent values (time series)
- **Interpolation** - Linear, polynomial, spline
- **KNN Imputation** - Use similar observations
- **Model-Based Imputation** - Predict missing values
- **Indicator Variables** - Flag for missingness

### E. Handling Outliers
- **Capping/Clipping** - Set upper/lower bounds
- **Winsorization** - Replace extremes with percentile values
- **Transformation** - Log, sqrt to reduce impact
- **Removal** - Drop extreme observations
- **Binning** - Group outliers into categories
- **Flagging** - Create binary outlier indicator

### F. Time-Based Features
- **Date Components** - Year, month, day, weekday, hour, quarter
- **Cyclical Encoding** - sin/cos for circular features
- **Elapsed Time** - Days/hours since event
- **Lag Features** - Previous time step values (t-1, t-2...)
- **Rolling Statistics** - Moving average, std, min, max
- **Time Differences** - Duration between events
- **Season/Holiday Flags** - Binary indicators
- **Trend Indicators** - Rolling slopes, changes

### G. Text Features
- **Bag of Words (BoW)** - Word frequency vectors
- **TF-IDF** - Term frequency-inverse document frequency
- **N-grams** - Bigrams, trigrams, character n-grams
- **Word Embeddings** - Word2Vec, GloVe, FastText
- **Transformer Embeddings** - BERT, RoBERTa
- **Text Statistics** - Length, word count, avg word length
- **Sentiment Scores** - Polarity, subjectivity
- **Part-of-Speech Tags** - Grammatical features
- **Named Entity Counts** - People, locations, organizations

### H. Geospatial Features
- **Distance Calculations** - Haversine, Euclidean
- **Clustering Coordinates** - K-means on lat/lon
- **Geohashing** - Location encoding
- **Proximity Features** - Distance to landmarks/POIs
- **Regional Aggregates** - Stats by area/region
- **Spatial Density** - Points within radius

### I. Feature Extraction (Dimensionality Reduction)
- **PCA** - Principal Component Analysis
- **t-SNE** - Non-linear visualization
- **UMAP** - Uniform Manifold Approximation
- **Autoencoders** - Neural network compression
- **LDA** - Linear Discriminant Analysis (supervised)
- **SVD** - Singular Value Decomposition
- **Factor Analysis** - Identify latent variables
- **ICA** - Independent Component Analysis

### J. Feature Selection

#### Filter Methods
- **Correlation Analysis** - Remove highly correlated features
- **Variance Threshold** - Remove low variance features
- **Statistical Tests** - Chi-square, ANOVA, Mutual Information
- **VIF** - Variance Inflation Factor for multicollinearity

#### Wrapper Methods
- **Forward Selection** - Incrementally add features
- **Backward Elimination** - Incrementally remove features
- **Recursive Feature Elimination (RFE)** - Iterative removal

#### Embedded Methods
- **L1 Regularization (Lasso)** - Automatic feature selection
- **Tree-based Importance** - Random Forest, XGBoost scores
- **Ridge/ElasticNet** - Regularized regression

### K. Domain-Specific Features

#### Image Features
- **Pixel Statistics** - Mean, std, histograms
- **Edge Detection** - Sobel, Canny operators
- **Texture Features** - GLCM, LBP
- **Color Features** - RGB, HSV distributions
- **CNN Embeddings** - Transfer learning features

#### Signal/Audio Features
- **Fourier Transform** - Frequency domain
- **Wavelet Transform** - Time-frequency features
- **Spectral Features** - Power spectral density
- **Statistical Moments** - Skewness, kurtosis
- **MFCC** - Mel-Frequency Cepstral Coefficients

#### Graph Features
- **Node Degree** - Number of connections
- **Centrality** - Importance measures
- **PageRank** - Graph ranking algorithm
- **Clustering Coefficient** - Local connectivity
- **Graph Embeddings** - Node2Vec, GraphSAGE

---

## 4. Model Selection & Cross-Validation

### A. Cross-Validation Techniques
- **Train-Test Split** - Simple 70/30 or 80/20 split
- **K-Fold CV** - k splits, train on k-1, test on 1
- **Stratified K-Fold** - Maintains class distribution
- **Leave-One-Out CV (LOOCV)** - n splits for n samples
- **Leave-P-Out CV** - Generalization of LOOCV
- **Repeated K-Fold** - Multiple K-Fold runs
- **Group K-Fold** - Keep groups together
- **Time Series Split** - Sequential split (no shuffling)
- **Nested CV** - Inner CV for tuning, outer for evaluation
- **Bootstrapping** - Random sampling with replacement
- **Hold-Out Validation** - Single train/val/test split
- **Monte Carlo CV** - Repeated random sub-sampling

### B. Model Comparison Methods
- **Grid Search CV** - Exhaustive parameter search
- **Random Search CV** - Random parameter sampling
- **Bayesian Optimization** - Probabilistic approach
- **Genetic Algorithms** - Evolutionary optimization
- **Hyperband** - Early stopping for efficiency
- **Optuna / TPE** - Tree-structured Parzen Estimator
- **SMBO** - Sequential Model-Based Optimization
- **Halving Search** - Iterative halving methods

### C. Automated Machine Learning (AutoML)
- **Auto-Sklearn** - Scikit-learn based AutoML
- **TPOT** - Genetic programming optimization
- **H2O AutoML** - Scalable AutoML
- **AutoKeras** - Neural architecture search
- **Google AutoML** - Cloud-based AutoML
- **Azure AutoML** - Microsoft's AutoML service

### D. Ensemble & Model Blending
- **Model Averaging** - Average predictions
- **Voting Classifier/Regressor** - Majority/mean vote
- **Stacking** - Meta-model learns from base models
- **Blending** - Similar to stacking with holdout set
- **Bagging** - Bootstrap aggregation (Random Forest)
- **Boosting** - Sequential error correction (AdaBoost, XGBoost, LightGBM, CatBoost)
- **Weighted Averaging** - Combine with learned weights

### E. Statistical Model Selection
- **Likelihood Ratio Test** - Compare nested models
- **Chi-Square Test** - Categorical associations
- **F-Test / ANOVA** - Compare group means
- **Wald Test** - Parameter significance
- **AIC** - Akaike Information Criterion
- **BIC** - Bayesian Information Criterion
- **Deviance / Log-Likelihood** - Model fit measures

---

## 5. Training Techniques

### A. Learning Paradigms

#### Supervised Learning
- **Regression** - Linear, Polynomial, Ridge, Lasso, ElasticNet, SVR
- **Classification** - Logistic, SVM, Decision Tree, Random Forest, Naive Bayes, KNN

#### Unsupervised Learning
- **Clustering** - K-Means, Hierarchical, DBSCAN, GMM, Spectral
- **Dimensionality Reduction** - PCA, ICA, t-SNE, UMAP

#### Semi-Supervised Learning
- **Self-training** - Use model predictions as labels
- **Pseudo-labeling** - Label unlabeled data
- **Graph-based** - Propagate labels through graph

#### Reinforcement Learning
- **Q-Learning** - Value-based method
- **Deep Q-Networks (DQN)** - Neural Q-learning
- **Policy Gradient** - Direct policy optimization
- **Actor-Critic** - A2C, A3C, PPO, DDPG, SAC

#### Self-Supervised Learning
- **Contrastive Learning** - SimCLR, MoCo
- **Masked Modeling** - BERT, MAE
- **Autoencoder Pretraining** - Unsupervised feature learning

### B. Optimization Algorithms
- **Batch Gradient Descent** - Full dataset per update
- **Stochastic Gradient Descent (SGD)** - One sample per update
- **Mini-Batch Gradient Descent** - Batch of samples
- **Momentum** - Accelerate convergence
- **Nesterov Accelerated Gradient (NAG)** - Look-ahead momentum
- **Adagrad** - Adaptive learning rates
- **RMSProp** - Moving average of squared gradients
- **Adam / AdamW** - Adaptive moment estimation
- **Nadam** - Nesterov + Adam
- **Adadelta** - Extension of Adagrad
- **L-BFGS** - Quasi-Newton method
- **SGD with Warm Restarts** - Cosine annealing
- **Lookahead Optimizer** - Look-ahead step

### C. Regularization Techniques
- **L1 Regularization (Lasso)** - Sparse weights
- **L2 Regularization (Ridge)** - Penalize large weights
- **Elastic Net** - L1 + L2 combination
- **Dropout** - Randomly drop neurons
- **Early Stopping** - Stop when validation loss increases
- **Batch Normalization** - Normalize layer inputs
- **Layer Normalization** - Normalize across features
- **Weight Decay** - L2 penalty on weights
- **Data Augmentation** - Increase training data variety
- **Label Smoothing** - Soften one-hot labels

### D. Advanced Training Strategies
- **Curriculum Learning** - Easy to hard examples
- **Transfer Learning** - Pretrained model fine-tuning
- **Multi-Task Learning** - Learn multiple tasks jointly
- **Meta-Learning** - Learn to learn
- **Few-Shot / One-Shot / Zero-Shot** - Learn from few examples
- **Active Learning** - Select informative samples
- **Online Learning** - Incremental updates
- **Federated Learning** - Distributed privacy-preserving
- **Distributed Training** - Multi-GPU/multi-node
- **Continual Learning** - Lifelong learning without forgetting
- **Knowledge Distillation** - Teacher-student models

### E. Deep Learning Optimization

#### Learning Rate Scheduling
- **Step Decay** - Reduce at fixed intervals
- **Exponential Decay** - Exponential reduction
- **Cyclic LR** - Oscillate between bounds
- **OneCycle LR** - Single cycle with peak
- **Warmup Scheduler** - Gradually increase LR

#### Weight Initialization
- **Xavier (Glorot)** - For tanh/sigmoid
- **He Initialization** - For ReLU activations
- **LeCun Initialization** - For SELU
- **Orthogonal Initialization** - Preserve gradient flow

#### Architecture-Specific Training

**CNN Training**
- Transfer Learning (VGG, ResNet, EfficientNet, Vision Transformers)
- Fine-tuning pretrained models

**RNN/LSTM/GRU Training**
- Truncated Backpropagation Through Time (TBPTT)
- Gradient Clipping to prevent explosion

**Transformer Training**
- Masked Language Modeling (BERT)
- Causal Language Modeling (GPT)
- Sequence-to-Sequence training

**GAN Training**
- Adversarial Training (Generator vs Discriminator)
- WGAN (Wasserstein GAN)
- Conditional GAN, StyleGAN, CycleGAN

**Diffusion Models**
- Denoising Score Matching
- Variational Diffusion Processes

### F. Data Handling During Training
- **Random Sampling** - Shuffle training data
- **Stratified Sampling** - Maintain class distribution
- **Oversampling** - SMOTE, ADASYN for minority class
- **Undersampling** - Reduce majority class
- **Weighted Loss Functions** - Class weights
- **Dynamic Data Loading** - DataLoader pipelines

### G. Parallel & Distributed Training
- **Data Parallelism** - Split data across devices
- **Model Parallelism** - Split model across devices
- **Pipeline Parallelism** - Layer-wise distribution
- **Tensor Parallelism** - Split tensors across devices
- **Mixed Precision Training** - FP16/BF16 for speed
- **Distributed Data Parallel (DDP)** - PyTorch DDP
- **Parameter Server** - Centralized parameter storage
- **Frameworks** - Horovod, DeepSpeed, Ray Train

### H. Specialized Training
- **Contrastive Learning** - SimCLR, MoCo
- **Masked Pretraining** - MAE, BERT
- **RLHF** - Reinforcement Learning from Human Feedback
- **Generative Pretraining** - GPT series
- **Domain Adaptation** - Transfer across domains
- **Quantization-Aware Training** - Train for quantization

---

## 6. Hyperparameter Tuning

### Core Methods
- **Grid Search** - Exhaustive search over grid
- **Random Search** - Random parameter sampling

### Advanced Methods
- **Bayesian Optimization** - Optuna, Hyperopt
- **Tree-structured Parzen Estimator (TPE)** - Probabilistic model
- **Sequential Model-Based Optimization (SMBO)** - Surrogate models
- **Genetic Algorithms** - Evolutionary search

### Resource-Efficient Methods
- **Hyperband** - Successive halving with adaptive allocation
- **ASHA** - Asynchronous Successive Halving
- **BOHB** - Bayesian Optimization + Hyperband

### Deep Learning Specific
- **Population-Based Training (PBT)** - Evolve hyperparameters
- **Learning Rate Finder** - Optimal LR discovery
- **Keras Tuner** - TensorFlow/Keras tuning
- **Ray Tune** - Scalable hyperparameter tuning

### Automated ML
- **Auto-Sklearn** - Automated scikit-learn
- **TPOT** - Genetic programming for ML
- **AutoKeras** - Neural architecture search

---

## 7. Model Evaluation

### A. Classification Metrics

#### Performance Metrics
- **Accuracy** - Correct predictions / Total predictions
- **Precision** - TP / (TP + FP)
- **Recall (Sensitivity/TPR)** - TP / (TP + FN)
- **Specificity (TNR)** - TN / (TN + FP)
- **F1 Score** - Harmonic mean of precision and recall
- **Fβ Score** - Weighted F-score (F0.5, F2)
- **Balanced Accuracy** - Average of recall per class
- **Hamming Loss** - Fraction of incorrect labels
- **Jaccard Index** - Intersection over union
- **Zero-One Loss** - Misclassification rate

#### Probabilistic Metrics
- **Log Loss / Cross-Entropy** - Penalize wrong probabilities
- **Brier Score** - Mean squared error of probabilities
- **KL Divergence** - Distance between distributions
- **Expected Calibration Error (ECE)** - Calibration measure
- **Top-K Accuracy** - Correct class in top K predictions

#### Ranking & Threshold Metrics
- **ROC-AUC** - Area under ROC curve
- **PR-AUC** - Area under Precision-Recall curve
- **Lift and Gain Chart** - Model improvement over random
- **CAP Curve** - Cumulative Accuracy Profile
- **Cohen's Kappa** - Agreement adjusted for chance
- **Matthews Correlation Coefficient (MCC)** - Balanced measure
- **G-Mean** - Geometric mean of sensitivity and specificity

#### Multi-Class & Multi-Label
- **Macro F1** - Unweighted average per class
- **Micro F1** - Global calculation across classes
- **Weighted F1** - Weighted by support
- **One-vs-Rest Evaluation** - Binary evaluation per class
- **Confusion Matrix** - Detailed error breakdown
- **Per-Class Metrics** - Individual class performance
- **LRAP** - Label Ranking Average Precision

### B. Regression Metrics

#### Error-Based Metrics
- **MAE** - Mean Absolute Error
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAPE** - Mean Absolute Percentage Error
- **MSLE** - Mean Squared Logarithmic Error
- **Median Absolute Error** - Robust to outliers
- **SMAPE** - Symmetric MAPE
- **Huber Loss** - Combined MAE and MSE
- **Quantile Loss** - Quantile regression loss

#### Goodness-of-Fit
- **R² Score** - Coefficient of determination
- **Adjusted R²** - Adjusted for number of features
- **Explained Variance Score** - Proportion explained

#### Information Criteria
- **AIC** - Akaike Information Criterion
- **BIC** - Bayesian Information Criterion
- **Deviance** - Goodness of fit measure
- **Log-Likelihood** - Probability of data given model

### C. Clustering Metrics

#### Internal Metrics (No labels)
- **Silhouette Score** - Cohesion and separation
- **Calinski-Harabasz Index** - Ratio of dispersions
- **Davies-Bouldin Index** - Average similarity measure
- **Inertia (WCSS)** - Within-cluster sum of squares

#### External Metrics (With labels)
- **Adjusted Rand Index (ARI)** - Similarity measure
- **Normalized Mutual Information (NMI)** - Information overlap
- **Homogeneity Score** - Same cluster same class
- **Completeness Score** - Same class same cluster
- **V-Measure** - Harmonic mean of homogeneity and completeness
- **Fowlkes-Mallows Index** - Geometric mean of precision and recall

### D. Time Series Metrics
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error
- **MAPE** - Mean Absolute Percentage Error
- **SMAPE** - Symmetric MAPE
- **MFE** - Mean Forecast Error
- **MASE** - Mean Absolute Scaled Error
- **Theil's U Statistic** - Forecast accuracy measure
- **CRPS** - Continuous Ranked Probability Score
- **Coverage Probability** - Prediction interval coverage
- **PICP** - Prediction Interval Coverage Probability

### E. Deep Learning Metrics
- **Loss Function Evaluation** - Training/validation loss
- **Perplexity** - Language model quality
- **BLEU** - Machine translation quality
- **ROUGE** - Summarization quality
- **METEOR** - Translation with synonyms
- **BERTScore** - Semantic similarity
- **CIDEr / SPICE** - Image captioning metrics
- **WER** - Word Error Rate (speech)
- **CER** - Character Error Rate
- **FID** - Fréchet Inception Distance (GANs)
- **Inception Score** - GAN quality
- **PSNR / SSIM** - Image quality metrics

### F. NLP & LLM Metrics
- **Perplexity** - Language model uncertainty
- **BLEU** - Translation quality
- **ROUGE** - Summarization overlap
- **METEOR** - Alignment-based translation
- **BERTScore** - Contextual embeddings similarity
- **GLEU** - Google BLEU variant
- **ChrF / TER** - Character/Translation Error Rate
- **Exact Match (EM)** - Exact string match
- **F1 for Token Overlap** - Token-level F1
- **Embedding Similarity** - Cosine/Euclidean distance
- **Human Evaluation** - Manual quality assessment
- **Toxicity / Bias Metrics** - Safety and fairness

### G. Post-Training Evaluation
- **Residual Analysis** - Error distribution analysis
- **Learning Curves** - Bias-variance tradeoff
- **Validation Curves** - Hyperparameter impact
- **Feature Importance** - Feature contribution
- **Permutation Importance** - Shuffle-based importance
- **Partial Dependence Plots (PDPs)** - Feature effect visualization
- **SHAP** - Shapley Additive Explanations
- **LIME** - Local Interpretable Model-agnostic Explanations
- **Fairness Evaluation** - Demographic parity, equal opportunity
- **Model Calibration** - Probability calibration curve
- **Drift Detection** - Data/concept drift monitoring
- **Robustness Testing** - Adversarial examples, noise

### H. Statistical Tests
- **Paired t-test** - Compare two models on same data
- **Wilcoxon Signed-Rank Test** - Non-parametric paired test
- **McNemar's Test** - Compare classifier errors
- **Cochran's Q Test** - Compare multiple classifiers
- **ANOVA / MANOVA** - Compare multiple groups
- **Chi-Square Test** - Categorical independence
- **Kolmogorov-Smirnov Test** - Distribution comparison
- **Kruskal-Wallis Test** - Non-parametric ANOVA
- **Permutation Test** - Resampling-based testing

### I. Business Metrics
- **Cost-Sensitive Metrics** - Weight errors by cost
- **ROI-based Evaluation** - Return on investment
- **Precision at K / Recall at K** - Top-K performance
- **Customer Lifetime Value Accuracy** - CLV prediction
- **Uplift Metrics** - Treatment effect measurement
- **Conversion Rate Optimization** - Conversion improvement
- **Churn Prediction Accuracy** - Customer retention

---

## 8. Model Deployment

### A. Traditional Deployment
- **Flask API** - Lightweight Python web framework
- **FastAPI** - Modern, fast Python API framework
- **Django REST Framework** - Full-featured web framework
- **Streamlit** - Interactive data apps
- **Gradio** - ML demo interfaces

### B. Cloud Deployment
- **AWS** - SageMaker, Lambda, EC2, ECS
- **Azure** - Azure ML, Azure Functions
- **Google Cloud** - AI Platform, Cloud Functions
- **IBM Watson** - IBM Cloud AI services

### C. Containerization & Orchestration
- **Docker** - Container platform
- **Kubernetes (K8s)** - Container orchestration
- **Helm** - Kubernetes package manager
- **Docker Compose** - Multi-container applications

### D. CI/CD & Automation
- **GitHub Actions** - GitHub automation
- **Jenkins** - Open-source automation server
- **GitLab CI/CD** - GitLab built-in CI/CD
- **Azure DevOps** - Microsoft DevOps platform
- **CircleCI** - Cloud CI/CD platform

### E. Model Serving Frameworks
- **TensorFlow Serving** - TensorFlow model serving
- **TorchServe** - PyTorch model serving
- **MLflow Model Serving** - MLflow deployment
- **BentoML** - ML model serving framework
- **Seldon Core** - Kubernetes-native serving
- **KServe** - Serverless inference on Kubernetes
- **Triton Inference Server** - NVIDIA inference server

### F. Edge & Mobile Deployment
- **TensorFlow Lite** - Mobile and edge ML
- **ONNX Runtime** - Cross-platform inference
- **Core ML** - iOS ML framework
- **ML Kit** - Android ML SDK
- **OpenVINO** - Intel edge AI toolkit

### G. Serverless Deployment
- **AWS Lambda** - AWS serverless compute
- **Google Cloud Functions** - Google serverless
- **Azure Functions** - Azure serverless compute

### H. Model Management & Monitoring
- **MLflow** - ML lifecycle management
- **Weights & Biases (W&B)** - Experiment tracking
- **Neptune.ai** - Metadata store for ML
- **Comet.ml** - ML experiment management
- **Model Versioning** - Track model versions
- **A/B Testing** - Compare model performance
- **Performance Monitoring** - Track metrics in production
- **Drift Detection** - Monitor data/concept drift

---

## 9. Visualization Techniques

### Matplotlib - Core Plots
- **Line Plot** - `plot()` - Trends over time
- **Scatter Plot** - `scatter()` - Relationships between variables
- **Bar Chart** - `bar()`, `barh()` - Categorica







