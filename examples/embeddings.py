#!/usr/bin/env python3
"""
This script demonstrates all-MiniLM-L6-v2 (Sentence Transformer)
used for FEATURE EXTRACTION
"""

import logging

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleClassifier(nn.Module):
  def __init__(self, embedding_dim, num_classes):
    super().__init__()
    self.classifier = nn.Sequential(
      nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
    )

  def forward(self, embeddings):
    return self.classifier(embeddings)


def demonstrate_sentence_transformer_pipeline():
  """
  PIPELINE 1: all-MiniLM-L6-v2 as FEATURE EXTRACTOR

  Text → [Sentence Transformer] → Embeddings → [Simple Neural Net] → Predictions

  This is used in train_nn.py
  """
  logger.info('\n' + '=' * 60)
  logger.info('PIPELINE 1: all-MiniLM-L6-v2 as FEATURE EXTRACTOR')
  logger.info('=' * 60)

  # Sample texts
  texts = [
    'Free money! Click here now!',
    'Hi, how are you doing today?',
    "URGENT: You've won $1000!",
  ]

  logger.info('Step 1: Load Sentence Transformer (FEATURE EXTRACTOR)')
  # This model ONLY does feature extraction - it doesn't make predictions
  feature_extractor = SentenceTransformer('all-MiniLM-L6-v2')
  logger.info(f'Model loaded: {feature_extractor}')

  logger.info('\nStep 2: Convert text to embeddings')
  # This converts text to fixed-size vectors (384 dimensions)
  embeddings = feature_extractor.encode(texts)
  logger.info(f'Input texts: {len(texts)}')
  logger.info(f'Output embeddings shape: {embeddings.shape}')
  logger.info(f'Each text becomes a {embeddings.shape[1]}-dimensional vector')

  logger.info('\nStep 3: Create a separate classifier')

  classifier = SimpleClassifier(embedding_dim=384, num_classes=2)
  logger.info(f'Classifier created: {classifier}')

  logger.info('\nStep 4: Make predictions')
  # Convert embeddings to predictions
  with torch.no_grad():
    embeddings_tensor = torch.FloatTensor(embeddings)
    logits = classifier(embeddings_tensor)
    predictions = torch.softmax(logits, dim=1)

  logger.info(f'Predictions shape: {predictions.shape}')
  logger.info('Pipeline: Text → Embeddings → Predictions')

  # Show the two-step process
  logger.info('\n📋 SUMMARY - Two-Step Process:')
  logger.info('1. Sentence Transformer: Text → Fixed embeddings (384-dim)')
  logger.info('2. Neural Network: Embeddings → Class predictions')
  logger.info('✅ Used in: train_nn.py with config_nn.yaml')


if __name__ == '__main__':
  demonstrate_sentence_transformer_pipeline()
