import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import structlog
from omegaconf import DictConfig, OmegaConf

logger = structlog.get_logger(__name__)


class ExperimentTracker:
  """Unified experiment tracking across different model types."""

  def __init__(self, results_file: str = 'experiment_results.csv'):
    self.results_file = Path(results_file)
    self.results_dir = Path('experiment_results')
    self.results_dir.mkdir(exist_ok=True)

  def log_experiment(
    self,
    cfg: DictConfig,
    metrics: Dict[str, Any],
    model_type: str,
    model_path: Optional[Path] = None,
    training_time: Optional[float] = None,
  ) -> None:
    """Log experiment results to both CSV and detailed JSON."""

    # Extract key information
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Basic experiment info
    experiment_data = {
      'timestamp': timestamp,
      'model_type': model_type,
      'experiment_name': cfg.experiment.name,
      'model_class': cfg.model._target_,
      'dataset': cfg.dataset.name,
      'random_seed': cfg.experiment.random_seed,
    }

    # Add model-specific parameters
    if model_type == 'scikit':
      experiment_data.update(self._extract_scikit_params(cfg))
    elif model_type == 'neural_network':
      experiment_data.update(self._extract_nn_params(cfg))
    elif model_type == 'transformer':
      experiment_data.update(self._extract_transformer_params(cfg))

    # Add metrics
    experiment_data.update(
      {
        'test_accuracy': metrics.get('accuracy', 0.0),
        'precision_macro': metrics.get('report', {})
        .get('macro avg', {})
        .get('precision', 0.0),
        'recall_macro': metrics.get('report', {})
        .get('macro avg', {})
        .get('recall', 0.0),
        'f1_macro': metrics.get('report', {}).get('macro avg', {}).get('f1-score', 0.0),
        'model_path': str(model_path) if model_path else '',
        'training_time_seconds': training_time,
      }
    )

    # Save to CSV (for easy comparison)
    self._save_to_csv(experiment_data)

    # Save detailed results to JSON
    detailed_data = {
      **experiment_data,
      'full_config': OmegaConf.to_container(cfg),
      'detailed_metrics': metrics,
    }
    self._save_detailed_json(detailed_data, timestamp)

    test_accuracy = experiment_data['test_accuracy']
    logger.info(f'Experiment logged: {model_type} - {test_accuracy:.4f} accuracy')

  def _extract_scikit_params(self, cfg: DictConfig) -> Dict[str, Any]:
    """Extract scikit-learn specific parameters."""
    params = {}

    # Model parameters
    if hasattr(cfg.model, 'C'):
      params['C'] = cfg.model.C
    if hasattr(cfg.model, 'kernel'):
      params['kernel'] = cfg.model.kernel
    if hasattr(cfg.model, 'alpha'):
      params['alpha'] = cfg.model.alpha

    # Feature extractor parameters (now directly under feature_extractor)
    if hasattr(cfg.feature_extractor, 'min_df'):
      params['min_df'] = cfg.feature_extractor.min_df
    if hasattr(cfg.feature_extractor, 'max_df'):
      params['max_df'] = cfg.feature_extractor.max_df
    if hasattr(cfg.feature_extractor, 'ngram_range'):
      params['ngram_range'] = str(cfg.feature_extractor.ngram_range)
    if hasattr(cfg.feature_extractor, 'max_features'):
      params['max_features'] = cfg.feature_extractor.max_features
    if hasattr(cfg.feature_extractor, 'stop_words'):
      params['stop_words'] = str(cfg.feature_extractor.stop_words)

    return params

  def _extract_nn_params(self, cfg: DictConfig) -> Dict[str, Any]:
    """Extract neural network specific parameters."""
    return {
      'hidden_units': cfg.model.get('hidden_units', 128),
      'learning_rate': cfg.model.get('learning_rate', 1e-3),
      'batch_size': cfg.model.get('batch_size', 32),
      'num_epochs': cfg.model.get('num_epochs', 5),
      'dropout_p': cfg.model.get('dropout_p', 0.5),
      'weight_decay': cfg.model.get('weight_decay', 1e-5),
      'embedding_model': cfg.feature_extractor.get(
        'model_name_or_path', 'all-MiniLM-L6-v2'
      ),
    }

  def _extract_transformer_params(self, cfg: DictConfig) -> Dict[str, Any]:
    """Extract transformer specific parameters."""
    return {
      'transformer_model': cfg.model.get('model_name', 'gpt2'),
      'learning_rate': cfg.model.get('learning_rate', 2e-5),
      'batch_size': cfg.model.get('batch_size', 16),
      'num_epochs': cfg.model.get('num_epochs', 3),
      'max_length': cfg.model.get('max_length', 512),
      'freeze_transformer': cfg.model.get('freeze_transformer', True),
      'dropout_p': cfg.model.get('dropout_p', 0.1),
    }

  def _save_to_csv(self, data: Dict[str, Any]) -> None:
    """Save experiment data to CSV for easy comparison."""
    df_new = pd.DataFrame([data])

    csv_path = self.results_dir / self.results_file
    if csv_path.exists():
      df_existing = pd.read_csv(csv_path)
      df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
      df_combined = df_new

    df_combined.to_csv(csv_path, index=False)

  def _save_detailed_json(self, data: Dict[str, Any], timestamp: str) -> None:
    """Save detailed experiment data to JSON."""
    json_path = self.results_dir / f'experiment_{timestamp}.json'
    with open(json_path, 'w') as f:
      json.dump(data, f, indent=2, default=str)

  def get_best_results(
    self, model_type: Optional[str] = None, metric: str = 'test_accuracy'
  ) -> pd.DataFrame:
    """Get best results, optionally filtered by model type."""
    csv_path = self.results_dir / self.results_file
    if not csv_path.exists():
      logger.warning('No experiment results found')
      return pd.DataFrame()

    df = pd.read_csv(csv_path)

    if model_type:
      df = df[df['model_type'] == model_type]

    if len(df) == 0:
      return df

    # Sort by metric and return top results
    return df.sort_values(metric, ascending=False)

  def compare_models(self, top_n: int = 5) -> pd.DataFrame:
    """Compare top performing models across all types."""
    csv_path = self.results_dir / self.results_file
    if not csv_path.exists():
      logger.warning('No experiment results found')
      return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Get top N results
    top_results = df.sort_values('test_accuracy', ascending=False).head(top_n)

    # Select key columns for comparison
    comparison_cols = [
      'timestamp',
      'model_type',
      'model_class',
      'test_accuracy',
      'precision_macro',
      'recall_macro',
      'f1_macro',
    ]

    return top_results[comparison_cols]
