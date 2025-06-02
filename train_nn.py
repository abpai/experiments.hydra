import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import structlog
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from load_dataset import load_dataset
from mlflow_integration import MLflowHydraIntegration

# Register tuple resolver for OmegaConf (to handle tuple parameters in configs)
OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))

logger = structlog.get_logger(__name__)


def log_device_info(device: torch.device) -> None:
  """Log information about the device being used."""
  if device.type == 'cuda':
    logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
  elif device.type == 'mps':
    logger.info('Using MPS')
  else:
    logger.info('Using CPU')


class EarlyStopping:
  """Early stopping utility to prevent overfitting."""

  def __init__(self, patience: int = 7, min_delta: float = 0.0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_score = None
    self.early_stop = False

  def __call__(self, val_score: float) -> bool:
    """Check if training should stop early."""
    if self.best_score is None:
      self.best_score = val_score
    elif val_score < self.best_score + self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = val_score
      self.counter = 0
    return self.early_stop


def train_epoch(
  model: torch.nn.Module,
  dataloader: DataLoader,
  optimizer: torch.optim.Optimizer,
) -> float:
  """Train the model for one epoch."""
  model.train()
  total_loss = 0
  num_batches = 0

  for batch_idx, (data, target) in enumerate(dataloader):
    try:
      # Model handles device placement automatically
      optimizer.zero_grad()
      loss = model.training_step((data, target))
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      num_batches += 1
    except Exception as e:
      logger.error(f'Error in training batch {batch_idx}: {str(e)}')
      continue

  return total_loss / max(num_batches, 1)


def evaluate(model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
  """Evaluate the model on the given dataloader."""
  model.eval()
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for data, target in dataloader:
      try:
        # Model handles device placement automatically
        output = model(data)
        preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
      except Exception as e:
        logger.error(f'Error in evaluation batch: {str(e)}')
        continue

  if not all_preds:
    logger.error('No valid predictions made during evaluation')
    return {'accuracy': 0.0, 'report': {}}

  accuracy = accuracy_score(all_targets, all_preds)
  try:
    report = classification_report(
      all_targets, all_preds, output_dict=True, zero_division=0
    )
  except Exception as e:
    logger.error(f'Error generating classification report: {str(e)}')
    report = {}

  return {'accuracy': accuracy, 'report': report}


def save_model(
  model: torch.nn.Module, cfg: DictConfig, metrics: Dict[str, Any]
) -> Optional[Path]:
  """Save the trained model with metadata."""
  try:
    # Create outputs directory if it doesn't exist
    output_dir = Path('outputs') / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp and accuracy
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    accuracy = metrics.get('accuracy', 0.0)
    model_name = cfg.model._target_.split('.')[-1]
    filename = f'{model_name}_{timestamp}_acc{accuracy:.4f}.pt'

    model_path = output_dir / filename

    # Save model state dict and metadata
    torch.save(
      {
        'model_state_dict': model.state_dict(),
        'model_config': OmegaConf.to_container(cfg.model),
        'metrics': metrics,
        'timestamp': timestamp,
      },
      model_path,
    )

    logger.info(f'Model saved to {model_path}')
    return model_path
  except Exception as e:
    logger.error(f'Failed to save model: {str(e)}')
    return None


def create_data_splits(
  X_text: np.ndarray, y_labels: np.ndarray, cfg: DictConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Create train/validation/test splits."""
  # First split: separate test set
  X_temp, X_test, y_temp, y_test = train_test_split(
    X_text,
    y_labels,
    test_size=cfg.dataset.test_split_ratio,
    random_state=cfg.experiment.random_seed,
    stratify=y_labels,
  )

  # Second split: separate train and validation from remaining data
  val_size = cfg.dataset.get('validation_split_ratio', 0.2)
  if val_size > 0:
    X_train, X_val, y_train, y_val = train_test_split(
      X_temp,
      y_temp,
      test_size=val_size,
      random_state=cfg.experiment.random_seed,
      stratify=y_temp,
    )
  else:
    X_train, X_val, y_train, y_val = X_temp, np.array([]), y_temp, np.array([])

  logger.info(
    f'Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}'
  )
  return X_train, X_val, X_test, y_train, y_val, y_test


@hydra.main(config_path='conf', config_name='config_nn', version_base=None)
def train_nn_pipeline(cfg: DictConfig) -> None:
  """Main training pipeline."""
  try:
    # Initialize MLflow tracking
    mlflow_integration = MLflowHydraIntegration(cfg)
    start_time = time.time()

    # Print configuration
    logger.info('--- HYDRA CONFIGURATION ---')
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info('-------------------------')

    # Set random seed for reproducibility
    torch.manual_seed(cfg.experiment.random_seed)
    np.random.seed(cfg.experiment.random_seed)

    # Start MLflow run and keep it active for the entire training
    mlflow_integration.start_run()

    # Log Hydra configuration to MLflow
    mlflow_integration.log_hydra_config()

    # Load data
    X_text, y_labels = load_dataset(cfg)

    # Create feature extractor
    logger.info('--- Initializing Feature Extractor ---')
    feature_extractor = hydra.utils.instantiate(
      {k: v for k, v in cfg.feature_extractor.items() if k != 'encode_params'}
    )

    # Get device info from feature extractor for logging
    device = next(feature_extractor.parameters()).device
    feature_extractor = feature_extractor.to(device)
    log_device_info(device)
    logger.info(f'Feature extractor moved to {device}')

    # Create train/val/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
      X_text, y_labels, cfg
    )

    # Log training metadata to MLflow
    mlflow_integration.log_training_metadata(
      train_size=len(X_train),
      val_size=len(X_val),
      test_size=len(X_test),
      device=str(device),
    )

    # Generate embeddings
    logger.info('Generating embeddings...')
    encode_kwargs = {
      'batch_size': cfg.feature_extractor.encode_params.batch_size,
      'convert_to_tensor': True,
      'device': device,
      'show_progress_bar': True,
    }

    X_train_embeddings = feature_extractor.encode(
      X_train.tolist() if isinstance(X_train, pd.Series) else X_train,
      **encode_kwargs,
    )

    X_test_embeddings = feature_extractor.encode(
      X_test.tolist() if isinstance(X_test, pd.Series) else X_test,
      **encode_kwargs,
    )

    X_val_embeddings = None
    if len(X_val) > 0:
      X_val_embeddings = feature_extractor.encode(
        X_val.tolist() if isinstance(X_val, pd.Series) else X_val,
        **encode_kwargs,
      )

    # Convert to tensors and create datasets
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_embeddings, y_train_tensor)
    test_dataset = TensorDataset(X_test_embeddings, y_test_tensor)

    val_dataset = None
    val_loader = None
    if X_val_embeddings is not None:
      y_val_tensor = torch.tensor(y_val, dtype=torch.long)
      val_dataset = TensorDataset(X_val_embeddings, y_val_tensor)
      val_loader = DataLoader(
        val_dataset, batch_size=cfg.model.batch_size, shuffle=False
      )

    # Create dataloaders
    train_loader = DataLoader(
      train_dataset, batch_size=cfg.model.batch_size, shuffle=True
    )
    test_loader = DataLoader(
      test_dataset, batch_size=cfg.model.batch_size, shuffle=False
    )

    # Initialize model
    logger.info('--- Initializing Model ---')
    cfg.model.embedding_dim = feature_extractor.get_sentence_embedding_dimension()
    logger.info(f'Setting model embedding_dim to {cfg.model.embedding_dim}')

    model = hydra.utils.instantiate(cfg.model)
    logger.info(f'Using model: {cfg.model._target_}')

    # Log device information (model handles device placement automatically)
    log_device_info(model.device)

    # Create optimizer
    optimizer = model.configure_optimizer()

    # Initialize early stopping if validation set exists
    early_stopping = None
    if val_loader is not None:
      patience = cfg.get('early_stopping_patience', 10)
      early_stopping = EarlyStopping(patience=patience)

    # Training loop
    logger.info('--- Training Model ---')
    num_epochs = cfg.model.num_epochs
    best_val_acc = 0.0

    for epoch in range(num_epochs):
      # Train
      train_loss = train_epoch(model, train_loader, optimizer)

      # Evaluate
      train_metrics = evaluate(model, train_loader)
      test_metrics = evaluate(model, test_loader)

      log_msg = (
        f'Epoch {epoch + 1}/{num_epochs} - '
        f'Train Loss: {train_loss:.4f}, '
        f'Train Acc: {train_metrics["accuracy"]:.4f}, '
        f'Test Acc: {test_metrics["accuracy"]:.4f}'
      )

      # Validation metrics if available
      if val_loader is not None:
        val_metrics = evaluate(model, val_loader)
        log_msg += f', Val Acc: {val_metrics["accuracy"]:.4f}'

        # Check early stopping
        if early_stopping(val_metrics['accuracy']):
          logger.info(f'Early stopping triggered at epoch {epoch + 1}')
          break

        if val_metrics['accuracy'] > best_val_acc:
          best_val_acc = val_metrics['accuracy']

      logger.info(log_msg)

      # Log epoch metrics to MLflow
      epoch_metrics = {
        'train_loss': train_loss,
        'train_accuracy': train_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
      }
      if val_loader is not None:
        epoch_metrics['val_accuracy'] = val_metrics['accuracy']

      mlflow_integration.log_metrics(epoch_metrics, step=epoch)

    # Final evaluation
    final_metrics = evaluate(model, test_loader)
    logger.info('--- Final Evaluation Results ---')
    logger.info(f'Test Accuracy: {final_metrics["accuracy"]:.4f}')

    # Print detailed classification report
    logger.info('Classification Report:')
    report = final_metrics.get('report', {})
    for class_name, metrics in report.items():
      if isinstance(metrics, dict) and 'precision' in metrics:
        log_msg = (
          f'Class {class_name}: '
          f'precision={metrics["precision"]:.4f}, '
          f'recall={metrics["recall"]:.4f}, '
          f'f1-score={metrics["f1-score"]:.4f}'
        )
        logger.info(log_msg)

    # Save model
    model_path = save_model(model, cfg, final_metrics)

    # Calculate training time and log final metrics to MLflow
    training_time = time.time() - start_time

    final_mlflow_metrics = {
      'final_test_accuracy': final_metrics['accuracy'],
      'training_time_seconds': training_time,
    }

    # Add macro-averaged metrics if available
    report = final_metrics.get('report', {})
    if 'macro avg' in report:
      macro_avg = report['macro avg']
      final_mlflow_metrics.update(
        {
          'precision_macro': macro_avg.get('precision', 0.0),
          'recall_macro': macro_avg.get('recall', 0.0),
          'f1_macro': macro_avg.get('f1-score', 0.0),
        }
      )

    mlflow_integration.log_metrics(final_mlflow_metrics)

    # End MLflow run
    mlflow_integration.end_run()

    if model_path:
      logger.info(f'Training completed successfully. Model saved to {model_path}')
    logger.info('MLflow tracking completed')

  except Exception as e:
    logger.error(f'Training failed with error: {str(e)}')
    raise


if __name__ == '__main__':
  train_nn_pipeline()
