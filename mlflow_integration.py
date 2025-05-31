from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import structlog
from omegaconf import DictConfig, OmegaConf

logger = structlog.get_logger(__name__)


class MLflowHydraIntegration:
  """Integration layer between MLflow and Hydra experiments."""

  def __init__(self, cfg: DictConfig):
    self.cfg = cfg
    self.run = None
    self.setup_mlflow()
    self.setup_autolog()

  def setup_mlflow(self):
    """Configure MLflow tracking."""
    # Set tracking URI to local directory
    tracking_uri = 'file:./mlruns'
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f'MLflow tracking URI: {tracking_uri}')

    # Set or create experiment
    experiment_name = self.cfg.experiment.name
    try:
      experiment = mlflow.get_experiment_by_name(experiment_name)
      if experiment is None:
        logger.info(f'Created new MLflow experiment: {experiment_name}')
      else:
        logger.info(f'Using existing MLflow experiment: {experiment_name}')

      mlflow.set_experiment(experiment_name)
    except Exception as e:
      logger.warning(f'Could not set MLflow experiment: {e}')

  def setup_autolog(self):
    """Enable appropriate autologging based on model type."""
    model_target = self.cfg.model._target_

    if 'sklearn' in model_target:
      mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
        silent=True,  # Reduce noise in logs
      )
      logger.info('MLflow sklearn autolog enabled')
    elif 'torch' in model_target or 'models.' in model_target:
      mlflow.pytorch.autolog(log_models=True, log_every_n_epoch=1, silent=True)
      logger.info('MLflow PyTorch autolog enabled')
    else:
      # Generic autolog for other frameworks
      mlflow.autolog(silent=True)
      logger.info('MLflow generic autolog enabled')

  def start_run(self, run_name: Optional[str] = None):
    """Start MLflow run with Hydra metadata."""
    # Create run name from config if not provided
    if not run_name:
      model_name = self.cfg.model._target_.split('.')[-1]
      dataset_name = self.cfg.dataset.name
      run_name = f'{model_name}_{dataset_name}'

    # Extract Hydra output directory
    try:
      from hydra.core.hydra_config import HydraConfig

      hydra_output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
      hydra_output_dir = 'unknown'

    # Start the run
    self.run = mlflow.start_run(
      run_name=run_name,
      tags={
        'model_class': self.cfg.model._target_,
        'dataset': self.cfg.dataset.name,
        'random_seed': str(self.cfg.experiment.random_seed),
        'hydra_output_dir': hydra_output_dir,
        'framework': self._get_framework_tag(),
        'hydra_config_name': getattr(self.cfg, '_name_', 'unknown'),
      },
    )

    logger.info(f'Started MLflow run: {run_name} (ID: {self.run.info.run_id})')
    return self.run

  def _get_framework_tag(self):
    """Determine framework from model target."""
    target = self.cfg.model._target_
    if 'sklearn' in target:
      return 'scikit-learn'
    elif 'models.embeddings' in target:
      return 'pytorch-embeddings'
    elif 'models.gpt2' in target:
      return 'pytorch-transformers'
    else:
      return 'unknown'

  def log_hydra_config(self):
    """Log complete Hydra configuration to MLflow."""
    if not mlflow.active_run():
      logger.warning('No active MLflow run. Cannot log configuration.')
      return

    try:
      # Log individual parameter groups for easy filtering
      self._log_model_params()
      self._log_dataset_params()
      self._log_feature_extractor_params()

      # Log full config as artifact
      self._log_config_artifact()

      logger.info('Hydra configuration logged to MLflow')
    except Exception as e:
      logger.error(f'Failed to log Hydra config to MLflow: {e}')

  def _log_model_params(self):
    """Log model parameters to MLflow."""
    model_params = {}
    for key, value in self.cfg.model.items():
      if not key.startswith('_'):
        # Convert complex types to strings
        if isinstance(value, (list, dict)):
          value = str(value)
        model_params[f'model.{key}'] = value

    if model_params:
      mlflow.log_params(model_params)

  def _log_dataset_params(self):
    """Log dataset parameters to MLflow."""
    dataset_params = {}
    for key, value in self.cfg.dataset.items():
      if not key.startswith('_'):
        if isinstance(value, (list, dict)):
          value = str(value)
        dataset_params[f'dataset.{key}'] = value

    if dataset_params:
      mlflow.log_params(dataset_params)

  def _log_feature_extractor_params(self):
    """Log feature extractor parameters to MLflow."""
    if not hasattr(self.cfg, 'feature_extractor'):
      return

    fe_params = {}
    for key, value in self.cfg.feature_extractor.items():
      if not key.startswith('_'):
        if isinstance(value, (list, dict)):
          value = str(value)
        fe_params[f'feature_extractor.{key}'] = value

    if fe_params:
      mlflow.log_params(fe_params)

  def _log_config_artifact(self):
    """Log full Hydra config as YAML artifact."""
    config_yaml = OmegaConf.to_yaml(self.cfg)

    # Create temporary file
    temp_path = Path('temp_hydra_config.yaml')
    try:
      with open(temp_path, 'w') as f:
        f.write(config_yaml)

      mlflow.log_artifact(str(temp_path), 'config')
    finally:
      # Clean up temp file
      if temp_path.exists():
        temp_path.unlink()

  def log_training_metadata(self, **kwargs):
    """Log additional training metadata."""
    if not mlflow.active_run():
      logger.warning('No active MLflow run. Cannot log training metadata.')
      return

    try:
      mlflow.log_params(kwargs)
    except Exception as e:
      logger.error(f'Failed to log training metadata: {e}')

  def log_metrics(self, metrics: dict, step: Optional[int] = None):
    """Log metrics to MLflow."""
    if not mlflow.active_run():
      logger.warning('No active MLflow run. Cannot log metrics.')
      return

    try:
      mlflow.log_metrics(metrics, step=step)
    except Exception as e:
      logger.error(f'Failed to log metrics: {e}')

  def log_evaluation_results(self, accuracy: float, report: dict, training_time: float):
    """Automatically log common evaluation metrics."""
    metrics = {
      'test_accuracy': accuracy,
      'training_time_seconds': training_time,
    }

    # Add macro-averaged metrics if available
    if 'macro avg' in report:
      macro_avg = report['macro avg']
      metrics.update(
        {
          'precision_macro': macro_avg.get('precision', 0.0),
          'recall_macro': macro_avg.get('recall', 0.0),
          'f1_macro': macro_avg.get('f1-score', 0.0),
        }
      )

    self.log_metrics(metrics)

  def end_run(self):
    """End the current MLflow run."""
    if self.run:
      mlflow.end_run()
      logger.info(f'Ended MLflow run: {self.run.info.run_id}')
      self.run = None

  def __enter__(self):
    """Context manager entry."""
    return self.start_run()

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit."""
    self.end_run()
