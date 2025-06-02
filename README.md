# Hydra ML Experiments

A comprehensive demonstration project showing how to use Hydra for configuration management in machine learning experiments. This project implements spam classification using three different approaches: scikit-learn models, neural networks, and transformer models.

## Overview

This project demonstrates:

- **Hydra Configuration Management**: Dynamic configuration composition and overrides
- **Multiple ML Approaches**: Scikit-learn, PyTorch neural networks, and transformer models
- **Modular Architecture**: Composable configurations for datasets, models, and feature extractors
- **Reproducible Experiments**: Consistent random seeding and output management
- **Production-Ready Code**: Error handling, logging, and model persistence

## Features

- 🔧 **Three Training Pipelines**: Scikit-learn, Neural Networks, and Transformers
- 📊 **Multiple Datasets**: SMS Spam and synthetic dummy data
- 🎯 **Various Models**: From Naive Bayes to BERT-based transformers
- 🔄 **Easy Experimentation**: Command-line configuration overrides
- 📈 **Comprehensive Evaluation**: Accuracy metrics and classification reports
- 💾 **Model Persistence**: Automatic model saving with metadata
- 📋 **MLflow Experiment Tracking**: Complete experiment tracking with MLflow
- 🖥️ **MLflow UI**: Interactive web interface for experiment comparison
- 📊 **Visual Analysis**: MLflow-powered plots and comparison tools

## Installation

### Prerequisites

- Python 3.12+
- uv package manager (recommended) or pip

### Setup

```shell
# Clone the repository
git clone <repository-url>
cd experiments.hydra

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Alternative: Install with pip
# pip install -e .
```

### Download the SMS spam dataset

```shell
python download_dataset.py
```

## Project Structure

```
experiments.hydra/
├── train_scikit.py             # Scikit-learn training pipeline
├── train_nn.py                 # Neural network training pipeline
├── train_llm.py                # LLM training pipeline
├── load_dataset.py             # Dataset loading utilities
├── mlflow_integration.py       # MLflow integration and configuration
├── analyze_experiments.py      # MLflow-based experiment analysis and visualization
├── launch_mlflow_ui.py         # MLflow UI launcher script
├── test_training_scripts.py    # Validation script for all training pipelines
├── models/                     # Model implementations
│   ├── embeddings.py           # Embeddings classifier model implementation
│   └── gpt2.py                 # GPT-2 classifier model implementation
├── conf/                       # Hydra configuration directory
│   ├── config_scikit.yaml      # Scikit-learn main config
│   ├── config_nn.yaml          # Neural network main config
│   ├── config_llm.yaml         # LLM main config
│   ├── model/                  # Model configurations
│   │   ├── embeddings_classifier.yaml
│   │   ├── gpt2_classifier.yaml
│   │   ├── logistic_regression.yaml
│   │   ├── naive_bayes.yaml
│   │   └── svm.yaml
│   ├── feature_extractor/      # Feature extraction configs
│   │   ├── tfidf_default.yaml
│   │   ├── tfidf_advanced.yaml
│   │   ├── count_vectorizer.yaml
│   │   └── sentence_transformer.yaml
│   └── dataset/                # Dataset configurations
│       ├── sms_spam.yaml
│       └── dummy_spam.yaml
├── data/                       # Dataset storage
├── outputs/                    # Training outputs and logs
├── mlruns/                     # MLflow experiment tracking data
└── pyproject.toml              # Project dependencies
```

## Usage

### 1. Scikit-learn Models (`train_scikit.py`)

Train traditional ML models with TF-IDF features:

```shell
# Default: Naive Bayes with TF-IDF
python train_scikit.py

# Try different models
python train_scikit.py model=logistic_regression
python train_scikit.py model=svm

# Override hyperparameters
python train_scikit.py model=logistic_regression model.C=0.1
python train_scikit.py model=svm model.kernel=rbf model.C=10.0

# Use different feature extractors
python train_scikit.py feature_extractor=tfidf_advanced
python train_scikit.py feature_extractor=count_vectorizer

# Change feature extraction parameters
python train_scikit.py feature_extractor.min_df=2
python train_scikit.py feature_extractor.max_df=0.8
python train_scikit.py feature_extractor.ngram_range=[1,3]

# Add new TF-IDF parameters (will be added to config)
python train_scikit.py +feature_extractor.max_features=5000
python train_scikit.py +feature_extractor.stop_words=english

# Use different dataset
python train_scikit.py dataset=dummy_spam
```

### 2. Neural Networks (`train_nn.py`)

Train PyTorch neural networks with sentence embeddings:

```shell
# Default: MLP with sentence transformers
python train_nn.py

# Override model architecture (embeddings_classifier is default)
python train_nn.py model=embeddings_classifier

# Adjust training parameters
python train_nn.py model.num_epochs=50 model.batch_size=64
python train_nn.py model.learning_rate=0.001 model.hidden_units=256

# Change sentence transformer model
python train_nn.py feature_extractor.model_name_or_path=all-distilroberta-v1

# Enable early stopping
python train_nn.py early_stopping_patience=10
```

### 3. Transformer LLM Models (`train_llm.py`)

Fine-tune transformer models for text classification:

```shell
# Default: GPT-2 classifier
python train_llm.py

# Adjust training parameters
python train_llm.py num_epochs=1 model.learning_rate=2e-5
python train_llm.py model.batch_size=16 model.max_length=256

# Freeze/unfreeze transformer layers
python train_llm.py model.freeze_transformer=true
python train_llm.py model.freeze_transformer=false

# Early stopping for fine-tuning
python train_llm.py early_stopping_patience=3
```

## Configuration System

### Main Configuration Files

Each training script uses a dedicated main configuration:

- **`config_scikit.yaml`**: Scikit-learn models with TF-IDF features
- **`config_nn.yaml`**: Neural networks with sentence embeddings
- **`config_llm.yaml`**: Transformer fine-tuning

### Available Models

#### Scikit-learn Models (`train_scikit.py`)

- `naive_bayes`: Multinomial Naive Bayes
- `logistic_regression`: Logistic Regression with L2 regularization
- `svm`: Support Vector Machine with linear kernel

#### Neural Network Models (`train_nn.py`)

- `embeddings_classifier`: Feedforward neural network with dropout regularization

#### Transformer Models (`train_llm.py`)

- `gpt2`: Fine-tunable GPT-2 with classification head

### Available Datasets

- `sms_spam`: Real SMS spam dataset
- `dummy_spam`: Synthetic dataset for testing

### Feature Extractors

#### For Scikit-learn Models

- `tfidf_default`: Basic TF-IDF vectorization
- `tfidf_advanced`: Advanced TF-IDF with stop words and bigrams
- `count_vectorizer`: Count-based feature extraction

#### For Neural Networks

- `sentence_transformer`: Sentence embeddings using transformers

#### Scikit-learn Feature Extractor Usage

All feature extractors now use Hydra's instantiation system for consistency:

```shell
# Use different feature extractors
python train_scikit.py feature_extractor=tfidf_default
python train_scikit.py feature_extractor=tfidf_advanced
python train_scikit.py feature_extractor=count_vectorizer

# Override specific parameters
python train_scikit.py feature_extractor.min_df=2
python train_scikit.py feature_extractor.max_df=0.8
python train_scikit.py feature_extractor.ngram_range=[1,3]

# Add new parameters with + prefix
python train_scikit.py +feature_extractor.max_features=5000
python train_scikit.py +feature_extractor.stop_words=english
```

#### Available TF-IDF Parameters

All sklearn.feature_extraction.text.TfidfVectorizer parameters are supported:

- `min_df`: Minimum document frequency (default: 1)
- `max_df`: Maximum document frequency (default: 1.0)
- `ngram_range`: N-gram range (default: [1,1])
- `max_features`: Maximum number of features
- `stop_words`: Stop words to remove ('english' or custom list)
- `lowercase`: Convert to lowercase (default: True)
- `strip_accents`: Remove accents ('unicode', 'ascii', or None)
- `binary`: Use binary term frequencies (default: False)

## Advanced Usage

### Hyperparameter Sweeps

Run multiple experiments with different configurations:

```shell
# Sweep over multiple models
python train_scikit.py -m model=naive_bayes,logistic_regression,svm

# Sweep over hyperparameters
python train_nn.py -m model.learning_rate=0.001,0.01,0.1 model.hidden_units=128,256,512

# Complex sweeps
python train_llm.py -m model.learning_rate=1e-5,2e-5,5e-5 model.batch_size=8,16,32
```

### Custom Configurations

Override any configuration parameter:

```shell
# Change experiment settings
python train_nn.py experiment.name=my_experiment experiment.random_seed=42

# Modify dataset parameters
python train_scikit.py dataset.test_split_ratio=0.3

# Adjust validation split
python train_nn.py dataset.validation_split_ratio=0.15
```

### Output Management

All experiments save outputs to timestamped directories:

```
outputs/
├── spam_classification_experiment/    # Scikit-learn outputs
├── nn_spam_classification/           # Neural network outputs
└── transformer_spam_classification/  # Transformer outputs
    └── 2024-01-15_14-30-45/         # Timestamped run
        ├── .hydra/                   # Hydra configs
        ├── train.log                 # Training logs
        └── models/                   # Saved models
```

## Experiment Tracking

The project uses **MLflow** for comprehensive experiment tracking and management.

### Automatic Tracking

Every training run is automatically tracked with MLflow:

- **Auto-logging**: Automatic model parameters, metrics, and artifacts
- **Web UI**: Interactive experiment comparison and visualization
- **Model Registry**: Built-in model versioning and deployment preparation
- **Artifact Storage**: Models, configs, and plots automatically saved
- **Run Comparison**: Side-by-side analysis of different experiments
- **Hyperparameter Tracking**: Complete parameter logging and analysis

**Data Storage**:

- `mlruns/` - MLflow experiment data and artifacts

### Analyzing Results

The analysis script automatically discovers all MLflow experiments and provides flexible analysis options. It handles missing data gracefully and provides helpful error messages when needed.

#### View Overall Summary

```shell
# Show summary of all experiments from MLflow (analyzes all experiments by default)
python analyze_experiments.py

# Example output:
# ================================================================================
# EXPERIMENT RESULTS SUMMARY
# ================================================================================
# Total experiments: 10
# Experiments: {'spam_classification_experiment': 7, 'nn_spam_classification': 2, 'transformer_spam_classification': 1}
# Frameworks: pytorch-transformers, pytorch-embeddings, scikit-learn
# Date range: 2024-01-15_10-30-45 to 2024-01-15_16-45-12
```

#### Focus on Specific Experiments

```shell
# Analyze a specific experiment
python analyze_experiments.py --experiment spam_classification_experiment
python analyze_experiments.py --experiment nn_spam_classification
python analyze_experiments.py --experiment transformer_spam_classification

# List available experiments (shown when invalid experiment name is provided)
python analyze_experiments.py --experiment invalid_name
```

#### Focus on Specific Model Types

```shell
# Analyze only scikit-learn models (across all experiments)
python analyze_experiments.py --model-type scikit

# Analyze neural networks
python analyze_experiments.py --model-type neural_network

# Analyze transformers
python analyze_experiments.py --model-type transformer
```

#### Generate Comparison Plots

```shell
# Generate comparison plots from MLflow data
python analyze_experiments.py --compare

# Save plots to file instead of displaying
python analyze_experiments.py --compare --save-plots model_comparison.png

# Combine specific experiment with plots
python analyze_experiments.py --experiment nn_spam_classification --compare
```

The comparison plots include:

- Accuracy distribution by framework
- Performance over time
- F1-score distributions
- Top 10 performing models
- Automatic handling of missing data (NaN values)

### MLflow UI (Recommended)

#### Launch MLflow Interface

```shell
# Start MLflow UI (recommended for viewing experiments)
python launch_mlflow_ui.py

# Or directly with MLflow
mlflow ui

# Access at: http://localhost:5000
```

#### MLflow Features

**Experiment Comparison**:

- Side-by-side model comparison
- Interactive metric plots
- Parameter vs. metric analysis
- Filter and search experiments

**Model Management**:

- Automatic model versioning
- Model artifact storage
- Deployment preparation
- Model lineage tracking

**Visualizations**:

- Learning curves (for neural networks)
- Hyperparameter importance
- Metric distributions
- Training progress plots

### Tracked Parameters

#### Scikit-learn Models

- Model type and hyperparameters (C, kernel, alpha)
- TF-IDF parameters (min_df, max_df, ngram_range)
- Feature extraction settings

#### Neural Networks

- Architecture (hidden_units, dropout_p)
- Training (learning_rate, batch_size, num_epochs)
- Embedding model (sentence transformer)
- Regularization (weight_decay)

#### Transformers

- Model name and size
- Fine-tuning parameters (learning_rate, freeze_transformer)
- Tokenization (max_length)
- Training settings (batch_size, num_epochs)

### Experiment Workflow Examples

#### Run and Compare Multiple Models

```shell
# Run experiments with different models (all tracked in MLflow)
python train_scikit.py model=naive_bayes
python train_scikit.py model=logistic_regression
python train_nn.py model.hidden_units=256
python train_llm.py model.freeze_transformer=false

# View results in MLflow UI (recommended)
python launch_mlflow_ui.py

# Or analyze with MLflow-based tools (analyzes all experiments by default)
python analyze_experiments.py --compare

# Focus on specific experiment results
python analyze_experiments.py --experiment spam_classification_experiment
```

#### Hyperparameter Sweeps with Tracking

```shell
# Run parameter sweeps - all automatically tracked in MLflow
python train_scikit.py -m model=naive_bayes,logistic_regression,svm
python train_nn.py -m model.learning_rate=0.001,0.01,0.1 model.hidden_units=64,128,256

# View best results for each model type from MLflow (searches all experiments)
python analyze_experiments.py --model-type scikit
python analyze_experiments.py --model-type neural_network

# Or focus on specific experiment sweeps
python analyze_experiments.py --experiment nn_spam_classification --compare
```

#### Find Best Performing Models

```shell
# View top performers across all experiments and frameworks from MLflow
python analyze_experiments.py

# The summary shows:
# TOP 5 PERFORMING MODELS
# --------------------------------------------------
# timestamp           framework      test_accuracy  experiment_name                f1_macro
# 2024-01-15_14-30-45 scikit-learn   0.9919        spam_classification_experiment  0.9821
# 2024-01-15_13-15-22 pytorch-embeddings 0.9901     nn_spam_classification          0.9788
# 2024-01-15_12-45-10 scikit-learn   0.9910        spam_classification_experiment  0.9806

# Focus on specific experiment's best models
python analyze_experiments.py --experiment spam_classification_experiment
```

### MLflow Data Structure

MLflow automatically organizes experiment data by training pipeline. Example:

```
mlruns/
├── 0/                              # Default experiment (usually empty)
├── 445719440628902944/             # spam_classification_experiment (scikit-learn)
├── 699778182175416388/             # nn_spam_classification (neural networks)
├── 170258548737220159/             # transformer_spam_classification (transformers)
│   ├── meta.yaml                   # Experiment metadata
│   └── <run_id>/                   # Individual run directories
│       ├── meta.yaml               # Run metadata
│       ├── metrics/                # Logged metrics (test_accuracy, f1_macro, etc.)
│       ├── params/                 # Logged parameters (model config, hyperparameters)
│       ├── tags/                   # Run tags (framework, dataset, etc.)
│       └── artifacts/              # Saved artifacts (models, configs, plots)
└── models/                         # MLflow model registry
```

### MLflow Tracked Data

All experiments automatically log:

| Data Type         | Description                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| **Metrics**       | test_accuracy, precision_macro, recall_macro, f1_macro                                           |
| **Parameters**    | Model hyperparameters and config values                                                          |
| **Tags**          | framework (scikit-learn, pytorch-embeddings, pytorch-transformers), dataset, experiment metadata |
| **Artifacts**     | Saved models, configuration files                                                                |
| **System Info**   | Git commit, Python version, packages                                                             |
| **Hardware Info** | CPU/GPU details, memory usage                                                                    |

### Integration with Hydra Sweeps

MLflow tracking works seamlessly with Hydra's multirun feature:

```shell
# Run sweep - each configuration automatically tracked in MLflow
python train_nn.py -m model.learning_rate=0.001,0.01 model.hidden_units=64,128 model.dropout_p=0.3,0.5

# Analyze sweep results via MLflow UI or analysis tools
python launch_mlflow_ui.py
python analyze_experiments.py --model-type neural_network
```

Each sweep configuration creates a separate MLflow run, making it easy to compare hyperparameter combinations and identify optimal settings.

## Model Performance

[ TBD ]

## Adding New Components

### Adding a New Model

1. **For Scikit-learn**: Create `conf/model/new_model.yaml`:

```yaml
# @package model
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
random_state: ${experiment.random_seed}
```

2. **For Neural Networks**: Add model class to `models/` directory and create config:

```yaml
# @package model
_target_: models.embeddings.CustomClassifier
embedding_dim: ${feature_extractor.embedding_dim}
hidden_units: 512
num_of_classes: 2
```

3. **For Transformers**: Create transformer config:

```yaml
# @package model
_target_: models.TransformerClassifier
model_name: "bert-base-uncased"
num_classes: 2
freeze_transformer: true
```

### Adding a New Dataset

Create `conf/dataset/new_dataset.yaml`:

```yaml
# @package dataset
_target_: load_dataset.load_custom_dataset
data_path: "data/custom_data.csv"
text_column: "message"
label_column: "category"
test_split_ratio: 0.2
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

   ```shell
   # Reduce batch size
   python train_llm.py model.batch_size=8
   python train_nn.py model.batch_size=32
   ```

2. **Slow Training**:

   ```shell
   # Use smaller models or reduce epochs
   python train_llm.py model.model_name=gpt2
   python train_nn.py model.num_epochs=10
   ```

3. **Import Errors**:

   ```shell
   # Ensure all dependencies are installed
   uv sync
   # Or reinstall specific packages
   uv add torch transformers sentence-transformers
   ```

4. **Configuration Errors**:

   If you encounter configuration-related issues, Hydra provides built-in debugging tools:

   ```shell
   # Check the final composed configuration before running
   python train_nn.py --cfg job

   # View configuration structure and validate overrides
   python train_scikit.py --cfg job model=svm

   # Check specific configuration groups
   python train_llm.py --cfg job model=gpt2

   # Validate configuration without running the experiment
   python train_nn.py --cfg job --help
   ```

   This helps identify:

   - Missing required parameters
   - Invalid configuration combinations
   - Typos in parameter names
   - Incorrect override syntax

5. **Hydra Override Errors**:

   If you get "Key 'X' is not in struct" errors:

   ```shell
   # Wrong: Trying to override non-existent parameter
   python train_scikit.py feature_extractor.params.max_features=5000

   # Correct: Add new parameter with + prefix
   python train_scikit.py +feature_extractor.params.max_features=5000

   # Or override existing parameters
   python train_scikit.py feature_extractor.params.min_df=2
   ```

   **Rule**: Use `+` prefix when adding new parameters that don't exist in the config.

### Performance Tips

- **GPU Usage**: Models automatically detect and use GPU when available
- **Memory Management**: Adjust batch sizes based on available memory
- **Early Stopping**: Use validation sets to prevent overfitting
- **Reproducibility**: Set consistent random seeds across experiments
- **Experiment Tracking**: All runs are automatically tracked for easy performance comparison

## Development

### Testing Training Scripts

Validate that all training scripts work correctly:

```shell
# Test all training pipelines with different configurations
python tests/training_scripts.py

# This tests:
# - Scikit-learn with different feature extractors
# - Neural networks with reduced epochs
# - Transformers with reduced epochs
# - Various parameter overrides
```

### Running Tests

```shell
# Install dev dependencies
uv sync --group dev

# Run tests
pytest tests/

# Run specific test categories
pytest tests/ -m smoke
```

### Code Quality

```shell
# Format code
ruff format .

# Lint code
ruff check .
```

## Dependencies

Key dependencies managed via `pyproject.toml`:

- **Core**: `hydra-core`, `omegaconf`, `structlog`
- **ML**: `torch`, `transformers`, `sentence-transformers`, `scikit-learn`
- **Tracking**: `mlflow` (experiment tracking and model registry)
- **Data**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Utilities**: `tqdm`, `pydantic`

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

For questions or issues, please open a GitHub issue.
