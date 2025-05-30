#!/usr/bin/env python3
"""
This script demonstrates all-MiniLM-L6-v2 (Sentence Transformer)
used for FEATURE EXTRACTION
"""

import logging

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2Classifier(nn.Module):
  def __init__(self, model_name: str = 'gpt2', num_of_classes: int = 2, **kwargs):
    super().__init__()
    self.model_name = model_name
    self.config = GPT2Config.from_pretrained(model_name)
    self.config.num_labels = num_of_classes
    self.gpt2 = GPT2Model.from_pretrained(model_name, config=self.config)
    self.classifier = nn.Linear(self.config.n_embd, num_of_classes)

    self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt2(input_ids, attention_mask=attention_mask, return_dict=True)

    last_hidden_state = outputs.last_hidden_state
    # -1 is last token
    # from: (batch_size, sequence_length, hidden_size)
    # to: (batch_size, hidden_size)
    last_token_hidden_state = last_hidden_state[:, -1, :]
    logits = self.classifier(last_token_hidden_state)
    return logits

def demonstrate_gpt2_pipeline():
  """
  PIPELINE 2: GPT-2 MODEL

  Text → [GPT-2 + Classification Head] → Predictions
  """
  logger.info('\n' + '=' * 60)
  logger.info('PIPELINE 2: GPT-2 as END-TO-END MODEL')
  logger.info('=' * 60)

  # Sample texts
  texts = [
    'Free money! Click here now!',
    'Hi, how are you doing today?',
    "URGENT: You've won $1000!",
  ]

  model_name = 'gpt2'
  num_classes = 2

  logger.info('\nStep 1: Initialize GPT2Classifier')
  model = GPT2Classifier(model_name, num_classes)

  logger.info('\nStep 2: Convert text to token ids')
  encoded_inputs = model.tokenizer(
    texts,
    padding=True,  # Pad to longest sequence in batch
    truncation=True,  # Truncate to model max length
    return_tensors='pt',
  )
  input_ids = encoded_inputs['input_ids']
  attention_mask = encoded_inputs['attention_mask']
  logger.info(f'Token IDs shape: {input_ids.shape}')

  logger.info('\nStep 3: Forward pass')
  logits = model(input_ids, attention_mask=attention_mask)
  logger.info(f'Logits shape: {logits.shape}')

  logger.info('\nStep 4: Predict')
  # Linear layer is not actually trained so
  # results are not actually meaningful
  predictions = torch.argmax(logits, dim=1)
  probabilities = torch.softmax(logits, dim=1)
  logger.info(f'Predictions: {predictions}')
  logger.info(f'Probabilities: {probabilities}')


if __name__ == '__main__':
  demonstrate_gpt2_pipeline()
