import os
from pathlib import Path

import pandas as pd
import requests
import structlog

logger = structlog.get_logger(__name__)

# Dataset URL
SMS_SPAM_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'


def download_sms_spam_dataset():
  """Download the SMS Spam Collection dataset from UCI ML Repository."""
  # Create data directory if it doesn't exist
  data_dir = Path('data')
  data_dir.mkdir(exist_ok=True)

  # Target files
  zip_path = data_dir / 'smsspamcollection.zip'
  csv_path = data_dir / 'spam.csv'

  # Skip if the CSV already exists
  if csv_path.exists():
    logger.info(f'Dataset already exists at {csv_path}')
    return csv_path

  # Download the dataset
  logger.info(f'Downloading SMS Spam Collection dataset from {SMS_SPAM_URL}')

  try:
    response = requests.get(SMS_SPAM_URL)
    response.raise_for_status()

    with open(zip_path, 'wb') as f:
      f.write(response.content)

    logger.info(f'Dataset downloaded to {zip_path}')

    # Extract and process the dataset
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(data_dir)

    # The extracted file is named 'SMSSpamCollection'
    raw_file = data_dir / 'SMSSpamCollection'

    # Convert to CSV
    if raw_file.exists():
      # The file has no header and is tab-separated
      # Format: <label>\t<text>
      with open(raw_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()

      # Create DataFrame
      data = []
      for line in lines:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
          label, text = parts
          data.append([label, text])

      df = pd.DataFrame(data, columns=['v1', 'v2'])

      # Save to CSV
      df.to_csv(csv_path, index=False)
      logger.info(f'Processed dataset saved to {csv_path}')

      # Clean up temporary files
      if zip_path.exists():
        os.remove(zip_path)
      if raw_file.exists():
        os.remove(raw_file)

      return csv_path
    else:
      logger.error(f'Extracted file not found: {raw_file}')
      return None

  except Exception as e:
    logger.error(f'Error downloading dataset: {str(e)}')
    return None


if __name__ == '__main__':
  dataset_path = download_sms_spam_dataset()

  if dataset_path:
    # Preview the dataset
    df = pd.read_csv(dataset_path)
    logger.info('\nDataset preview:')
    logger.info(f'Total samples: {len(df)}')
    logger.info(f'Class distribution: {df["v1"].value_counts().to_dict()}')
    logger.info('\nFirst 5 samples:')
    logger.info(df.head())
  else:
    logger.error('Failed to download dataset.')
