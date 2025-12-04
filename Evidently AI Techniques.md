# Evidently AI Techniques Reference

A comprehensive guide to all important techniques and concepts in Evidently AI for ML model monitoring and evaluation.

## Table of Contents

- [Core Monitoring Techniques](#core-monitoring-techniques)
- [Test Suites & Presets](#test-suites--presets)
- [ML-Specific Techniques](#ml-specific-techniques)
- [Getting Started](#getting-started)

---

## Core Monitoring Techniques

### Data Drift Detection

Monitors changes in input data distributions over time. Essential for detecting when your production data starts to differ from training data.

**Statistical Tests Used:**
- Kolmogorov-Smirnov test
- Jensen-Shannon divergence
- Wasserstein distance
- Chi-squared test
- Population Stability Index (PSI)

**Use Case:** Detect when feature distributions shift, indicating your model may need retraining.

---

### Data Quality Monitoring

Tracks the health and integrity of your data pipeline.

**Key Checks:**
- Missing values detection
- Duplicate records identification
- Out-of-range values
- Schema changes
- Type mismatches
- Constant values

**Use Case:** Catch data pipeline issues before they affect model predictions.

---

### Model Performance Degradation

Monitors prediction quality metrics over time when ground truth is available.

**Classification Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Log loss

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared

**Use Case:** Track actual model performance in production and trigger retraining when performance drops.

---

### Prediction Drift

Detects changes in model output distributions without requiring ground truth labels.

**Key Features:**
- Monitors prediction probabilities
- Detects shifts in predicted class distributions
- Works when labels arrive with delay

**Use Case:** Early warning system when ground truth isn't immediately available.

---

### Target Drift

Monitors changes in the actual target variable distribution (concept drift).

**Key Aspects:**
- Detects shifts in label distributions
- Identifies concept drift scenarios
- Compares reference vs current targets

**Use Case:** Understand if the underlying patterns in your data are changing.

---

## Test Suites & Presets

### Data Stability Tests

Validates that data characteristics remain consistent between reference and current datasets.

**Includes:**
- Feature value ranges
- Statistical properties
- Categorical value sets
- Correlation patterns

---

### NoTargetPerformance Preset

For scenarios where ground truth labels aren't immediately available.

**Components:**
- Prediction drift detection
- Data drift monitoring
- Data quality checks

**Use Case:** Monitor models in production before labels become available.

---

### DataQuality Preset

Comprehensive data quality checks bundled together.

**Automated Checks:**
- All missing value patterns
- Duplicate detection
- Type validation
- Range checks

**Use Case:** Quick setup for data validation pipelines.

---

### DataDrift Preset

Quick setup for drift detection across all features.

**Features:**
- Automatic feature selection
- Multiple statistical tests
- Visual drift reports
- Drift score aggregation

**Use Case:** One-line drift monitoring for entire datasets.

---

## ML-Specific Techniques

### Classification Performance

Specialized metrics and visualizations for classification tasks.

**Includes:**
- Confusion matrix analysis
- Per-class metrics
- Threshold optimization
- Calibration curves
- Lift and gain charts

---

### Regression Performance

Tailored metrics for regression models.

**Key Metrics:**
- Error distribution analysis
- Residual plots
- Prediction vs actual scatter
- Error quantiles

---

### Ranking Performance

Metrics for recommendation and ranking systems.

**Includes:**
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Hit Rate

---

### Text Descriptors

For NLP models and text-based features.

**Monitoring Capabilities:**
- Text length distribution
- Vocabulary drift
- Sentiment analysis
- Out-of-vocabulary (OOV) words
- Special character patterns
- Language detection

**Use Case:** Monitor text inputs to NLP models for distribution shifts.

---

## Getting Started

### Installation

```bash
pip install evidently
```

### Basic Usage Example

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd

# Load your data
reference_data = pd.read_csv('reference.csv')
current_data = pd.read_csv('current.csv')

# Create report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

# Run the report
report.run(reference_data=reference_data, current_data=current_data)

# Save results
report.save_html('monitoring_report.html')
```

### Test Suite Example

```python
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfMissingValues

# Create test suite
tests = TestSuite(tests=[
    TestColumnDrift(column_name='feature_1'),
    TestShareOfMissingValues(column_name='feature_2', lt=0.05)
])

# Run tests
tests.run(reference_data=reference_data, current_data=current_data)

# Check results
tests.save_html('test_results.html')
```

---

## Best Practices

1. **Start with Presets** - Use built-in presets for quick setup, customize later
2. **Define Reference Data Carefully** - Use stable, representative data as your baseline
3. **Set Appropriate Thresholds** - Tune drift sensitivity based on your use case
4. **Monitor Continuously** - Set up automated monitoring pipelines
5. **Combine Multiple Techniques** - Use drift + quality + performance together
6. **Act on Insights** - Create alerts and triggers for model retraining

---

## Resources

- [Official Documentation](https://docs.evidentlyai.com/)
- [GitHub Repository](https://github.com/evidentlyai/evidently)
- [Example Notebooks](https://github.com/evidentlyai/evidently/tree/main/examples)

---

## License

This reference guide is for educational purposes. Evidently AI is an open-source project with Apache 2.0 license.