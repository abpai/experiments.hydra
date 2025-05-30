import logging

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

# Suppress transformers warnings for cleaner output
logging.getLogger('transformers').setLevel(logging.ERROR)


class GPT2Classifier(nn.Module):
  """
  Real transformer-based classifier using GPT-2 architecture.

  Uses a pre-trained GPT-2 model with a custom classification head.
  The transformer layers can be frozen or fine-tuned.
  """

  def __init__(
    self,
    model_name: str = 'gpt2',  # Can be "gpt2", "gpt2-medium", etc.
    num_of_classes: int = 2,
    dropout_p: float = 0.1,
    freeze_transformer: bool = True,
    max_length: int = 512,
    learning_rate: float = 2e-5,  # Lower LR typical for transformer fine-tuning
    weight_decay: float = 0.01,
    batch_size: int = 16,  # Smaller batch size for memory efficiency
    num_epochs: int = 3,  # Fewer epochs typical for fine-tuning
    **kwargs,
  ):
    super().__init__()

    # Input validation
    if num_of_classes <= 0:
      raise ValueError('Number of classes must be positive')
    if not 0 <= dropout_p <= 1:
      raise ValueError('Dropout probability must be between 0 and 1')
    if max_length <= 0:
      raise ValueError('Max length must be positive')

    # Load pre-trained GPT-2 model and tokenizer
    try:
      self.config = GPT2Config.from_pretrained(model_name)
      self.transformer = GPT2Model.from_pretrained(model_name)
      self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

      # Add padding token if it doesn't exist
      if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

    except Exception as e:
      raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    # Store parameters
    self.num_of_classes = num_of_classes
    self.max_length = max_length
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.freeze_transformer = freeze_transformer

    # Freeze transformer layers if specified
    if freeze_transformer:
      for param in self.transformer.parameters():
        param.requires_grad = False

    # Classification head
    hidden_size = self.config.n_embd
    self.classifier = nn.Sequential(
      nn.Dropout(dropout_p),
      nn.Linear(hidden_size, hidden_size // 2),
      nn.ReLU(),
      nn.Dropout(dropout_p),
      nn.Linear(hidden_size // 2, num_of_classes),
    )

    # Loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # Device selection and model placement
    self.device = self._get_device()
    self.to(self.device)

    # Cache for tokenized inputs to avoid re-tokenizing
    self._input_cache = {}

  def _get_device(self) -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
      return torch.device('cuda')
    elif torch.backends.mps.is_available():
      return torch.device('mps')
    else:
      return torch.device('cpu')

  def tokenize_texts(self, texts: list) -> dict[str, torch.Tensor]:
    """Tokenize input texts."""
    # Convert single text to list
    if isinstance(texts, str):
      texts = [texts]

    # Tokenize with padding and truncation
    tokenized = self.tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=self.max_length,
      return_tensors='pt',
    )

    return tokenized

  def forward(
    self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> torch.Tensor:
    """Forward pass through the transformer and classification head."""
    # Ensure tensors are on the correct device
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)

    # Get transformer outputs
    transformer_outputs = self.transformer(
      input_ids=input_ids, attention_mask=attention_mask
    )

    # Use the last hidden state of the last token (like GPT-2 for classification)
    last_hidden_states = transformer_outputs.last_hidden_state

    # Get the representation of the last non-padding token for each sequence
    # Find the last non-padding token for each sequence
    batch_size = input_ids.shape[0]
    sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
    last_token_hidden = last_hidden_states[torch.arange(batch_size), sequence_lengths]

    # Pass through classification head
    logits = self.classifier(last_token_hidden)

    return logits

  def forward_from_texts(self, texts: list) -> torch.Tensor:
    """Forward pass directly from text inputs."""
    tokenized = self.tokenize_texts(texts)
    input_ids = tokenized['input_ids'].to(self.device)
    attention_mask = tokenized['attention_mask'].to(self.device)

    return self.forward(input_ids, attention_mask)

  def configure_optimizer(self) -> torch.optim.Optimizer:
    """
    Configure optimizer with different learning rates for transformer and classifier.

    The transformer is frozen by default, so we use a lower learning rate for it.
    The classifier is trained with a higher learning rate.
    """
    # Different learning rates for transformer and classification head
    transformer_params = []
    classifier_params = []

    for name, param in self.named_parameters():
      if param.requires_grad:
        if 'transformer' in name:
          transformer_params.append(param)
        else:
          classifier_params.append(param)

    # Use lower learning rate for pre-trained transformer
    optimizer_params = [
      {'params': transformer_params, 'lr': self.learning_rate},
      {
        'params': classifier_params,
        'lr': self.learning_rate * 10,
      },  # Higher LR for new layers
    ]

    return optim.AdamW(optimizer_params, weight_decay=self.weight_decay)

  def training_step(
    self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  ) -> torch.Tensor:
    """Compute loss for a training batch."""
    input_ids, attention_mask, labels = batch

    # Move batch to device
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    labels = labels.to(self.device)

    logits = self.forward(input_ids, attention_mask)
    loss = self.loss_fn(logits, labels)

    return loss

  def predict(self, texts: list) -> torch.Tensor:
    """Generate predictions for input texts."""
    self.eval()
    with torch.no_grad():
      logits = self.forward_from_texts(texts)
      return torch.argmax(logits, dim=1)

  def predict_proba(self, texts: list) -> torch.Tensor:
    """Generate class probabilities for input texts."""
    self.eval()
    with torch.no_grad():
      logits = self.forward_from_texts(texts)
      return torch.softmax(logits, dim=1)

  def get_model_info(self) -> dict[str, any]:
    """Get information about the model."""
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    return {
      'model_name': self.config.name_or_path
      if hasattr(self.config, 'name_or_path')
      else 'gpt2',
      'total_parameters': total_params,
      'trainable_parameters': trainable_params,
      'frozen_transformer': self.freeze_transformer,
      'max_length': self.max_length,
      'num_classes': self.num_of_classes,
    }
