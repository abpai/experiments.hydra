import time

import hydra
import numpy as np
import pandas as pd
import structlog
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from experiment_tracker import ExperimentTracker
from load_dataset import load_dataset
from mlflow_integration import MLflowHydraIntegration

# Register tuple resolver for OmegaConf (to handle tuple parameters in configs)
OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))

logger = structlog.get_logger(__name__)


@hydra.main(config_path='conf', config_name='config_scikit', version_base=None)
def train_model_pipeline(cfg: DictConfig) -> None:
  """Main training pipeline for scikit-learn models."""
  try:
    # Initialize both tracking systems
    tracker = ExperimentTracker()  # Keep for backwards compatibility
    mlflow_integration = MLflowHydraIntegration(cfg)
    start_time = time.time()

    # Print configuration
    logger.info('--- HYDRA CONFIGURATION ---')
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info('-------------------------')

    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.random_seed)

    # Start MLflow run
    with mlflow_integration:
      # Log Hydra configuration to MLflow
      mlflow_integration.log_hydra_config()

      # Load data
      X_text, y_labels = load_dataset(cfg)

      # Create train/test splits
      X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y_labels,
        test_size=cfg.dataset.test_split_ratio,
        random_state=cfg.experiment.random_seed,
        stratify=y_labels,
      )
      logger.info('--- Data Split ---')
      logger.info(f'Train size: {len(X_train_text)}, Test size: {len(X_test_text)}')

      # Log training metadata to MLflow
      mlflow_integration.log_training_metadata(
        train_size=len(X_train_text),
        test_size=len(X_test_text),
        train_test_ratio=f'{len(X_train_text)}:{len(X_test_text)}',
      )

      # Initialize feature extractor using Hydra
      logger.info('--- Initializing Feature Extractor ---')
      feature_extractor = hydra.utils.instantiate(cfg.feature_extractor)
      logger.info(f'Feature extractor: {cfg.feature_extractor._target_}')

      # Generate features
      logger.info('Generating features...')
      X_train_vec = feature_extractor.fit_transform(X_train_text)
      X_test_vec = feature_extractor.transform(X_test_text)
      logger.info(f'Shape of training features: {X_train_vec.shape}')
      logger.info(f'Shape of testing features: {X_test_vec.shape}')

      # Log feature extraction metadata
      mlflow_integration.log_training_metadata(
        n_features=X_train_vec.shape[1],
        feature_density=X_train_vec.nnz / X_train_vec.shape[0] / X_train_vec.shape[1],
      )

      # Initialize model using Hydra
      logger.info('--- Initializing Model ---')
      model = hydra.utils.instantiate(cfg.model)
      logger.info(f'Using model: {cfg.model._target_}')

      # Train model (MLflow will auto-log model parameters and metrics)
      logger.info('--- Training Model ---')
      model.fit(X_train_vec, y_train)
      logger.info('Training complete.')

      # Evaluate model
      predictions = model.predict(X_test_vec)
      accuracy = accuracy_score(y_test, predictions)

      # Generate detailed classification report
      try:
        report = classification_report(
          y_test, predictions, output_dict=True, zero_division=0
        )
      except Exception as e:
        logger.error(f'Error generating classification report: {str(e)}')
        report = {}

      # Calculate training time
      training_time = time.time() - start_time

      # Final evaluation
      final_metrics = {'accuracy': accuracy, 'report': report}
      logger.info('--- Final Evaluation Results ---')
      logger.info(f'Test Accuracy: {final_metrics["accuracy"]:.4f}')

      # Log metrics to MLflow
      mlflow_metrics = {
        'test_accuracy': accuracy,
        'training_time_seconds': training_time,
      }

      # Add macro-averaged metrics if available
      if 'macro avg' in report:
        macro_avg = report['macro avg']
        mlflow_metrics.update(
          {
            'precision_macro': macro_avg.get('precision', 0.0),
            'recall_macro': macro_avg.get('recall', 0.0),
            'f1_macro': macro_avg.get('f1-score', 0.0),
          }
        )

      mlflow_integration.log_metrics(mlflow_metrics)

      # Print detailed classification report
      logger.info('Classification Report:')
      for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
          log_msg = (
            f'Class {class_name}: '
            f'precision={metrics["precision"]:.4f}, '
            f'recall={metrics["recall"]:.4f}, '
            f'f1-score={metrics["f1-score"]:.4f}'
          )
          logger.info(log_msg)

      # Log experiment results to custom tracker (for backwards compatibility)
      tracker.log_experiment(
        cfg=cfg, metrics=final_metrics, model_type='scikit', training_time=training_time
      )

      # Hydra output information
      run_dir = HydraConfig.get().runtime.output_dir
      logger.info('Training completed successfully.')
      logger.info(f'Output and logs saved to: {run_dir}')
      logger.info('Experiment tracked in experiment_results/')
      logger.info(f'MLflow run ID: {mlflow_integration.run.info.run_id}')

  except Exception as e:
    logger.error(f'Training failed with error: {str(e)}')
    raise


if __name__ == '__main__':
  train_model_pipeline()
