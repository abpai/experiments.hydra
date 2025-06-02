import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import structlog
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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


class TextClassificationDataset(Dataset):
  """Dataset for text classification with transformer tokenization."""

  def __init__(
    self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512
  ):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    label = self.labels[idx]

    # Tokenize text
    encoding = self.tokenizer(
      text,
      truncation=True,
      padding='max_length',
      max_length=self.max_length,
      return_tensors='pt',
    )

    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(label, dtype=torch.long),
    }


def create_data_loaders(
  train_texts: List[str],
  train_labels: List[int],
  val_texts: List[str],
  val_labels: List[int],
  test_texts: List[str],
  test_labels: List[int],
  tokenizer,
  cfg: DictConfig,
) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
  """Create data loaders for pre-split train, validation, and test sets."""

  logger.info(
    f'Train size: {len(train_texts)}, '
    f'Val size: {len(val_texts)}, '
    f'Test size: {len(test_texts)}'
  )

  # Create datasets
  max_length = cfg.model.max_length
  batch_size = cfg.model.batch_size

  train_dataset = TextClassificationDataset(
    train_texts, train_labels, tokenizer, max_length
  )
  test_dataset = TextClassificationDataset(
    test_texts, test_labels, tokenizer, max_length
  )

  val_dataset = None
  if len(val_texts) > 0:
    val_dataset = TextClassificationDataset(
      val_texts, val_labels, tokenizer, max_length
    )

  # Create data loaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  val_loader = (
    DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if val_dataset
    else None
  )

  return train_loader, val_loader, test_loader


def split_data_for_loaders(
  X_texts: np.ndarray, y_labels: np.ndarray, cfg: DictConfig
) -> tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
  """Split data into train/val/test sets according to config ratios."""

  # Create train/val/test splits
  X_temp, X_test, y_temp, y_test = train_test_split(
    X_texts,
    y_labels,
    test_size=cfg.dataset.test_split_ratio,
    random_state=cfg.experiment.random_seed,
    stratify=y_labels,
  )

  # Create validation split if specified
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

  # Convert to lists for consistency
  train_texts = X_train.tolist()
  train_labels = y_train.tolist()
  val_texts = X_val.tolist() if len(X_val) > 0 else []
  val_labels = y_val.tolist() if len(y_val) > 0 else []
  test_texts = X_test.tolist()
  test_labels = y_test.tolist()

  return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def train_epoch(
  model: torch.nn.Module,
  dataloader: DataLoader,
  optimizer: torch.optim.Optimizer,
) -> float:
  """Train the model for one epoch."""
  model.train()
  total_loss = 0
  num_batches = 0

  for batch in dataloader:
    try:
      # Model handles device placement automatically
      input_ids = batch['input_ids']
      attention_mask = batch['attention_mask']
      labels = batch['labels']

      optimizer.zero_grad()
      loss = model.training_step((input_ids, attention_mask, labels))
      loss.backward()

      # Gradient clipping for transformer fine-tuning
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      optimizer.step()

      total_loss += loss.item()
      num_batches += 1
    except Exception as e:
      logger.error(f'Error in training batch: {str(e)}')
      continue

  return total_loss / max(num_batches, 1)


def evaluate(model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
  """Evaluate the model on the given dataloader."""
  model.eval()
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for batch in dataloader:
      try:
        # Model handles device placement automatically
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
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


@hydra.main(config_path='conf', config_name='config_llm', version_base=None)
def train_transformer_pipeline(cfg: DictConfig) -> None:
  """Main training pipeline for transformer models."""
  try:
    # Initialize MLflow tracking
    mlflow_integration = MLflowHydraIntegration(cfg)
    start_time = time.time()

    # Start MLflow run and log configuration
    mlflow_integration.start_run()
    mlflow_integration.log_hydra_config()

    # Print configuration
    logger.info('--- HYDRA CONFIGURATION ---')
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info('-------------------------')

    # Set random seed for reproducibility
    torch.manual_seed(cfg.experiment.random_seed)
    np.random.seed(cfg.experiment.random_seed)

    # Load data
    X_text, y_labels = load_dataset(cfg)

    # Initialize model (this will also initialize the tokenizer)
    logger.info('--- Initializing Transformer Model ---')
    model = hydra.utils.instantiate(cfg.model)

    # Log device information (model handles device placement automatically)
    log_device_info(model.device)

    # Log model information
    model_info = model.get_model_info()
    logger.info(f'Model: {model_info["model_name"]}')
    logger.info(f'Total parameters: {model_info["total_parameters"]:,}')
    logger.info(f'Trainable parameters: {model_info["trainable_parameters"]:,}')
    logger.info(f'Frozen transformer: {model_info["frozen_transformer"]}')

    # Log model metadata to MLflow
    mlflow_integration.log_training_metadata(
      model_name=model_info['model_name'],
      total_parameters=model_info['total_parameters'],
      trainable_parameters=model_info['trainable_parameters'],
      frozen_transformer=model_info['frozen_transformer'],
    )

    # Split data according to config ratios
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
      split_data_for_loaders(X_text, y_labels, cfg)
    )

    # Create data loaders (using model's tokenizer)
    train_loader, val_loader, test_loader = create_data_loaders(
      train_texts,
      train_labels,
      val_texts,
      val_labels,
      test_texts,
      test_labels,
      model.tokenizer,
      cfg,
    )

    # Create optimizer
    optimizer = model.configure_optimizer()

    # Initialize early stopping if validation set exists
    early_stopping = None
    if val_loader is not None:
      patience = cfg.get(
        'early_stopping_patience', 5
      )  # Shorter patience for transformers
      early_stopping = EarlyStopping(patience=patience)

    # Training loop
    logger.info('--- Training Transformer Model ---')
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
  train_transformer_pipeline()
