from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import structlog
from omegaconf import DictConfig, OmegaConf

logger = structlog.get_logger(__name__)


def validate_dataset_config(cfg: DictConfig) -> None:
  """Validate dataset configuration parameters."""
  if not hasattr(cfg.dataset, 'name'):
    raise ValueError("Dataset configuration must specify 'name'")

  if cfg.dataset.name == 'sms_spam':
    required_fields = ['path', 'text_column', 'label_column']
    for field in required_fields:
      if not hasattr(cfg.dataset, field):
        raise ValueError(f"SMS spam dataset requires '{field}' in configuration")


def load_dummy_dataset() -> Tuple[np.ndarray, np.ndarray]:
  """Load the dummy spam dataset for testing purposes."""
  data = {
    'text': [
      'Free money offer, claim now!',
      "WINNER! You've won a $1000 cash prize.",
      'Meeting scheduled for tomorrow at 10am regarding project updates.',
      'Hello John, how are you doing today?',
      'Special discount on all products this weekend only!',
      'URGENT: Your account requires immediate attention for security reasons.',
      "Let's discuss the project deliverables next week.",
      'Please confirm your subscription to our newsletter.',
      'Buy one get one free, limited time offer.',
      "Don't forget our appointment next Tuesday at 3pm.",
    ],
    'label': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],  # 1 for spam, 0 for not spam (ham)
  }
  df = pd.DataFrame(data)
  logger.info(f'Loaded dummy dataset with {len(df)} samples')
  logger.info(f'Class distribution: {df["label"].value_counts().to_dict()}')

  return df['text'].values, df['label'].values


def load_sms_spam_dataset(cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
  """Load the SMS Spam Collection dataset."""
  dataset_path = Path(cfg.dataset.path)

  if not dataset_path.exists():
    raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

  # Try different encodings if latin-1 fails
  encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
  df = None

  for encoding in encodings:
    try:
      logger.info(f'Attempting to load dataset with {encoding} encoding')
      df = pd.read_csv(dataset_path, encoding=encoding)
      logger.info(f'Successfully loaded with {encoding} encoding')
      break
    except UnicodeDecodeError:
      logger.warning(f'Failed to load with {encoding} encoding')
      continue
    except Exception as e:
      logger.error(f'Error loading dataset with {encoding}: {str(e)}')
      continue

  if df is None:
    raise ValueError(f'Could not load dataset from {dataset_path} with any encoding')

  # Validate required columns exist
  text_column = cfg.dataset.text_column
  label_column = cfg.dataset.label_column

  if text_column not in df.columns:
    raise ValueError(
      f"Text column '{text_column}' not found in dataset. "
      f'Available columns: {list(df.columns)}'
    )
  if label_column not in df.columns:
    raise ValueError(
      f"Label column '{label_column}' not found in dataset. "
      f'Available columns: {list(df.columns)}'
    )

  # Handle missing values
  initial_count = len(df)
  df = df.dropna(subset=[text_column, label_column])
  if len(df) < initial_count:
    logger.warning(f'Dropped {initial_count - len(df)} rows with missing values')

  # Map text labels to numeric values if needed
  if hasattr(cfg.dataset, 'label_mapping') and cfg.dataset.label_mapping:
    original_labels = df[label_column].unique()
    df[label_column] = df[label_column].map(cfg.dataset.label_mapping)

    # Check for unmapped labels
    if df[label_column].isna().any():
      unmapped = df[df[label_column].isna()][label_column].unique()
      logger.warning(f'Found unmapped labels: {unmapped}')
      df = df.dropna(subset=[label_column])

    logger.info(
      f'Mapped labels: {dict(zip(original_labels, df[label_column].unique()))}'
    )

  # Validate label values
  unique_labels = df[label_column].unique()
  if not all(isinstance(label, (int, float)) for label in unique_labels):
    logger.warning(f'Non-numeric labels found: {unique_labels}')

  logger.info(f'Dataset loaded with {len(df)} samples')
  logger.info(f'Class distribution: {df[label_column].value_counts().to_dict()}')

  # Convert to numpy arrays
  texts = df[text_column].values
  labels = df[label_column].values

  # Ensure labels are integers
  try:
    labels = labels.astype(int)
  except ValueError as e:
    logger.error(f'Could not convert labels to integers: {e}')
    raise ValueError(f'Labels must be convertible to integers: {unique_labels}')

  return texts, labels


def load_dataset(cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
  """
  Load dataset based on configuration.

  Args:
      cfg: Configuration object containing dataset specifications

  Returns:
      Tuple of (texts, labels) as numpy arrays

  Raises:
      ValueError: If dataset configuration is invalid
      FileNotFoundError: If dataset file is not found
  """
  try:
    validate_dataset_config(cfg)

    logger.info(f'Loading dataset: {cfg.dataset.name}')

    if cfg.dataset.name == 'dummy_spam':
      return load_dummy_dataset()
    elif cfg.dataset.name == 'sms_spam':
      return load_sms_spam_dataset(cfg)
    else:
      raise ValueError(f'Unknown dataset: {cfg.dataset.name}')

  except Exception as e:
    logger.error(f'Error loading dataset: {str(e)}')
    if cfg.dataset.name != 'dummy_spam':
      logger.warning('Falling back to dummy dataset')
      return load_dummy_dataset()
    else:
      raise
