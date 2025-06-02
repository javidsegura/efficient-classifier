# Efficient Classifier

[![PyPI version](https://badge.fury.io/py/efficient-classifier.svg)](https://pypi.org/project/efficient-classifier/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, dataset-agnostic machine learning framework for rapid development and deployment of classification pipelines on tabular data. Advanced DevOps tools.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Models & Metrics](#supported-models--metrics)
- [Use Cases](#use-cases)
- [Performance](#performance)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)

## Overview

Efficient Classifier is an enterprise-grade machine learning framework designed to accelerate the development lifecycle from data preprocessing to model deployment. Built with scalability and reproducibility in mind, it provides a unified interface for experimenting with multiple classification pipelines while maintaining rigorous tracking of experiments and results.

The framework supports both binary and multiclass classification tasks and has been extensively validated on real-world datasets, including the CCCS-CIC-AndMal-2020 cybersecurity dataset where it achieved 92% F1-score performance.

### Research & Validation

Our framework has been applied to cutting-edge cybersecurity research:

- **[Research Paper](https://drive.google.com/drive/folders/1GksAEhtbiqzj-pGVJixrn35E6DRu44gK?usp=drive_link)** - CCCS-CIC-AndMal-2020 Analysis
- **[Complete Results](https://drive.google.com/drive/folders/1Ui2EmIr-5rrXPkab1lGquHp_cQ7w14yA?usp=sharing)** - Plots, logs, and execution history
- **[Technical Report](https://docs.google.com/document/d/1yH9gvnJVSH9GLv9ATQ5JQWA2z8Jy4umxxRfMF-y2fiU/edit?usp=drive_link)** - Methodology and findings
- **[EDA Notebook](https://drive.google.com/file/d/1NbvUQKDtAbgVKoTZ2rG1YcpiZOrNB8Gq/view?usp=sharing)** - Exploratory data analysis
- **[Presentation](https://www.canva.com/design/DAGnoUCnQmQ/VgZLdpPD2IpRFxJj_7TuLg/edit?utm_content=DAGnoUCnQmQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)** - Project overview

## Key Features

### 🚀 **Rapid Pipeline Development**
- Multi-pipeline orchestration with customizable architectures
- Zero-boilerplate configuration through YAML
- Automated hyperparameter optimization (Grid, Random, Bayesian)
- One-command execution from data to deployment

### 🔬 **Advanced Analytics & Visualization**
- Comprehensive residual analysis and confusion matrices
- LIME-based feature importance with permutation testing
- Model calibration with reliability diagrams
- Cross-validation with stratified sampling
- Real-time training progress monitoring

### 🛠 **Production-Ready DevOps**
- Slack bot integration for real-time notifications
- Automated DAG visualization of pipeline architectures
- Model serialization with joblib/pickle support
- Comprehensive logging and experiment tracking
- Built-in testing framework integration

### ⚡ **High-Performance Computing**
- Multithreaded processing where parallelization is beneficial
- Memory-efficient data handling for large datasets
- Optimized feature selection algorithms (Boruta, L1 regularization)
- Smart caching mechanisms for repeated operations

## Architecture

The framework follows a modular, stage-based architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loading  │ -> │  Preprocessing   │ -> │ Feature Analysis│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 |
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     DevOps      │ <- │    Modeling      │ <- │   Evaluation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Stage-Specific Capabilities

| Stage | Capability | Description |
|-------|------------|-------------|
| **Data Management** | Smart Splitting | Adaptive train/validation/test splits with distribution analysis |
| | Distribution Validation | Statistical tests ensuring consistent feature distributions across splits |
| **Preprocessing** | Advanced Encoding | One-hot encoding with automatic categorical detection |
| | Intelligent Imputation | Multiple strategies for handling missing values |
| | Outlier Detection | IQR and percentile-based detection with configurable treatment |
| | Robust Scaling | StandardScaler, RobustScaler, and MinMaxScaler support |
| | Class Balancing | SMOTE and ADASYN implementations for imbalanced datasets |
| **Feature Engineering** | Automated Selection | Mutual information, variance filtering, and multicollinearity detection |
| | Advanced Techniques | Boruta feature selection and L1 regularization |
| | Custom Engineering | Dataset-specific feature creation hooks |
| **Modeling** | Ensemble Methods | Stacked generalization with configurable base learners |
| | Neural Networks | Feed-forward architectures with epoch-wise monitoring |
| | Model Comparison | Cross-model evaluation with statistical significance testing |
| **DevOps** | Real-time Monitoring | Slack integration for training progress and alerts |
| | Experiment Tracking | Comprehensive CSV logging with metadata |
| | Visualization | Automated DAG generation for pipeline architecture |

## Installation

### PyPI Installation (Recommended)

```bash
pip install efficient-classifier
```

### Development Installation

```bash
git clone https://github.com/javidsegura/efficient-classifier.git
cd efficient-classifier
pip install -r requirements.txt
```

### Environment Setup

For Slack bot integration, create a `.env` file:

```bash
SLACK_BOT_TOKEN=your_bot_token
SLACK_SIGNING_SECRET=your_signing_secret
SLACK_APP_TOKEN=your_app_token
```

## Quick Start

### Basic Usage

```python
from efficient_classifier import PipelineManager

# Initialize with configuration
manager = PipelineManager('configurations.yaml')

# Execute complete pipeline
results = manager.run_all_pipelines()

# Access best model
best_model = results.get_best_model()
predictions = best_model.predict(X_test)
```

### Custom Dataset Integration

1. **Configure dataset-specific cleaning** in `pipeline_runner.py`:
```python
def _clean_dataset_set_up_dataset_specific(self, df):
    # Your custom preprocessing logic
    return cleaned_df
```

2. **Implement feature engineering** in `featureAnalysis_runner.py`:
```python
def _run_feature_engineering_dataset_specific(self, df):
    # Your custom feature engineering
    return engineered_df
```

3. **Update boundary conditions** in `bound_config.py` for data validation.

## Configuration

The framework uses a comprehensive YAML configuration system. Key configuration sections:

### Pipeline Definition
```yaml
general:
  pipelines_names: ["baseline", "advanced", "ensemble"]
  max_plots_per_function: 10  # Control visualization output
```

### Data Processing
```yaml
phase_runners:
  dataset_runners:
    split_df:
      p: [0.7, 0.8, 0.9]  # Split ratios to evaluate
      step: 0.05          # Granularity of split analysis
    encoding:
      y_column: "target"  # Target variable name
```

### Model Configuration
```yaml
modelling_runner:
  class_weights:
    weights: {0: 1.0, 1: 2.0}  # Handle class imbalance
  models_to_include:
    baseline: ["Random Forest", "Logistic Regression"]
    advanced: ["XGBoost", "Neural Network"]
  optimization:
    method: "bayesian"  # grid, random, or bayesian
    cv_folds: 5
```

For complete configuration options, see the [detailed documentation](documentation/library_detailed.md).

## Supported Models & Metrics

### Machine Learning Models

**Tree-Based Algorithms:**
- Random Forest, Decision Trees, Gradient Boosting
- XGBoost, LightGBM, CatBoost
- AdaBoost with configurable base estimators

**Linear Models:**
- Logistic Regression, Ridge Classifier
- Linear/Non-linear SVM, SGD Classifier
- Elastic Net with L1/L2 regularization

**Advanced Methods:**
- Feed-Forward Neural Networks
- Ensemble Stacking (meta-learning)
- K-Nearest Neighbors, Gaussian Naive Bayes

**Baseline Models:**
- Majority Class Classifier for benchmarking

### Evaluation Metrics

- **Classification Accuracy** - Overall correctness
- **Precision, Recall, F1-Score** - Class-specific performance
- **Cohen's Kappa** - Inter-rater reliability
- **Weighted Accuracy** - Class-imbalance adjusted accuracy
- **ROC-AUC** - Area under receiver operating characteristic
- **Calibration Metrics** - Reliability diagrams and Brier score

### Adding Custom Models

Extend model support by modifying `modelling_runner.py`:

```python
def _model_initializers(self):
    models = {
        # Existing models...
        "Custom Model": YourCustomClassifier(
            param1=self.config['custom_param']
        )
    }
    return models
```

## Use Cases

### MANTIS: Cybersecurity Threat Detection

Our flagship application demonstrates the framework's capabilities in cybersecurity:

**Dataset:** CCCS-CIC-AndMal-2020 (Android malware detection)
**Performance:** 92% F1-score with Random Forest + Stacking ensemble
**Scale:** 200,000+ samples with 464 features
**Deployment:** Production-ready model with 15ms inference time

**Key Results:**
- Outperformed baseline approaches by 23%
- Identified 847 critical features through automated selection
- Achieved 99.1% precision for malware detection

### Benchmark Datasets

**Titanic Survival Prediction:** [View Results](https://drive.google.com/drive/folders/1ALECwX7EgQa3XgQLHtkjvAcKo2_XFIA7?usp=sharing)
- 89.3% accuracy with ensemble methods
- Comprehensive feature engineering pipeline

**Iris Classification:** [View Results](https://drive.google.com/drive/folders/1zzUIgnC4K44kmkDQ3j3zV9qeyJgybqPr?usp=drive_link)
- 97.8% accuracy across all pipeline configurations
- Validation of multi-class capabilities

## Performance

### Benchmarks

| Dataset | Samples | Features | Best Model | F1-Score | Training Time |
|---------|---------|----------|------------|----------|---------------|
| CCCS-CIC-AndMal-2020 | 200K+ | 464 | Random Forest | 92.0% | 45 min |
| Titanic | 891 | 12 | Stacking Ensemble | 89.3% | 2 min |
| Iris | 150 | 4 | Neural Network | 97.8% | 30 sec |

### Optimization Features

- **Memory Management:** Efficient handling of datasets up to 1M+ rows
- **Parallel Processing:** Multi-core utilization for independent operations
- **Early Stopping:** Automatic convergence detection for iterative algorithms
- **Caching:** Intelligent result caching for repeated experiments

## Model Deployment

### Serialization & Inference

```python
# Save trained pipeline
manager.serialize_model(best_pipeline, 'production_model.pkl')

# Load for inference
loaded_model = manager.load_model('production_model.pkl')

# Production predictions
predictions = loaded_model.model_sklearn.predict(X_new)
probabilities = loaded_model.model_sklearn.predict_proba(X_new)
```

### Production Integration

The serialized models contain:
- Trained sklearn estimator objects
- Complete preprocessing pipelines
- Feature engineering transformations
- Model metadata and performance metrics

## Monitoring & Visualization

### Real-Time Notifications

![SlackBot Integration](https://github.com/user-attachments/assets/19045a75-32dc-4777-8cfb-e6e39ec4f073)

*Slack bot provides real-time updates on training progress, model performance, and system alerts.*

### Pipeline Visualization

![DAG Pipeline Visualizer](https://github.com/user-attachments/assets/b06781c6-b703-4695-a5c3-ea720809884d)

*Automatically generated DAG visualization showing pipeline architecture, data flow, and performance metrics.*

## Roadmap & Known Limitations

### Upcoming Features

- **Multi-label Classification Support**
- **Cyclical Feature Encoding** for temporal data
- **Cloud Deployment Integration** (AWS, GCP, Azure)
- **Docker Containerization** for production deployment
- **Advanced AutoML Capabilities** with neural architecture search

### Current Limitations

- **Missing Value Handling:** Assumes preprocessed data (manual handling required)
- **Grid Search Configuration:** Complex setup process for new parameter spaces
- **Stacking Visualization:** Not included in DAG visualization
- **Per-Pipeline Feature Selection:** Currently uses unified feature selection

### Performance Considerations

Operations marked as 'MAJOR IMPACT IN PERFORMANCE' in configuration:
- Bayesian optimization with large parameter spaces
- Neural network training with extensive architectures
- Cross-validation with high fold counts
- Feature selection on high-dimensional datasets

## Contributing

We welcome contributions from the community! Please follow these guidelines:

### Development Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation for API changes

### Pull Request Template

Please include:
- **Description** of changes and motivation
- **Testing** performed and results
- **Breaking Changes** if applicable
- **Documentation** updates

## Documentation

### Comprehensive Guides

- **[Library Architecture](documentation/library_detailed.md)** - Design decisions and implementation details
- **[API Reference](documentation/api_reference.md)** - Complete function and class documentation
- **[Configuration Guide](documentation/configuration.md)** - YAML parameter explanations
- **[Troubleshooting](documentation/troubleshooting.md)** - Common issues and solutions

### Research Publications

Access our peer-reviewed research and detailed technical reports through the links provided in the [Research & Validation](#research--validation) section.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{efficient_classifier_2024,
  title={Efficient Classifier: A Dataset-Agnostic ML Framework},
  author={[Javier D., Caterina B, Juan A., Federica C., Irina I., Juliette J.]},
  year={2025},
  url={https://github.com/javidsegura/efficient-classifier}
}
```

## Acknowledgments

- Built with scikit-learn, XGBoost, and other open-source ML libraries
- Validated on datasets from the Canadian Centre for Cyber Security
- Community contributors and beta testers

---

**Ready to accelerate your ML workflow?** Install via `pip install efficient-classifier` and check out our [Quick Start Guide](#quick-start).