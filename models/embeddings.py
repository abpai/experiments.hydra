import torch
import torch.nn as nn
import torch.optim as optim


class EmbeddingsClassifier(nn.Module):
  """
  Simple neural network classifier with flexible binary/multi-class support.

  Features:
  - Adaptive loss function selection based on number of classes
  - Dropout regularization
  - Configurable architecture
  """

  def __init__(
    self,
    embedding_dim: int,
    hidden_units: int,
    num_of_classes: int = 2,
    dropout_p: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    num_epochs: int = 5,
    **kwargs,
  ):
    super().__init__()

    # Input validation
    if embedding_dim <= 0 or hidden_units <= 0 or num_of_classes <= 0:
      raise ValueError('Dimensions must be positive integers')
    if not 0 <= dropout_p <= 1:
      raise ValueError('Dropout probability must be between 0 and 1')
    if learning_rate <= 0:
      raise ValueError('Learning rate must be positive')

    # Network layers
    self.linear1 = nn.Linear(embedding_dim, hidden_units)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=dropout_p)
    self.linear2 = nn.Linear(hidden_units, num_of_classes)

    # Store parameters for training
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.num_of_classes = num_of_classes

    # Loss function selection
    self.loss_fn = self._select_loss_function(num_of_classes)

    # Device selection and model placement
    self.device = self._get_device()
    self.to(self.device)

  def _get_device(self) -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
      return torch.device('cuda')
    elif torch.backends.mps.is_available():
      return torch.device('mps')
    else:
      return torch.device('cpu')

  def _select_loss_function(self, num_classes: int) -> nn.Module:
    """Select appropriate loss function based on number of classes."""
    if num_classes == 1:
      return nn.BCEWithLogitsLoss()
    elif num_classes == 2:
      return nn.CrossEntropyLoss()
    else:
      return nn.CrossEntropyLoss()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the network."""
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x

  def configure_optimizer(self) -> torch.optim.Optimizer:
    """Configure the optimizer for training."""
    return optim.Adam(
      self.parameters(),
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
    )

  def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Compute loss for a training batch."""
    x, y = batch
    x, y = x.to(self.device), y.to(self.device)
    logits = self(x)

    # Handle different loss function requirements
    if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
      if logits.dim() > 1 and logits.shape[1] == 1:
        logits = logits.squeeze()
      loss = self.loss_fn(logits, y.float())
    else:
      loss = self.loss_fn(logits, y)
    return loss

  def predict(self, x: torch.Tensor) -> torch.Tensor:
    """Generate predictions for input data."""
    self.eval()
    with torch.no_grad():
      x = x.to(self.device)
      logits = self(x)

      if self.num_of_classes == 1 or (logits.dim() > 1 and logits.shape[1] == 1):
        # Binary classification with single output
        return (torch.sigmoid(logits.squeeze()) > 0.5).long()
      else:
        # Multi-class classification
        return torch.argmax(logits, dim=1)

  def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
    """Generate class probabilities for input data."""
    self.eval()
    with torch.no_grad():
      x = x.to(self.device)
      logits = self(x)

      if self.num_of_classes == 1 or (logits.dim() > 1 and logits.shape[1] == 1):
        # Binary classification
        probs = torch.sigmoid(logits.squeeze())
        if probs.dim() == 0:
          probs = probs.unsqueeze(0)
        return torch.stack([1 - probs, probs], dim=1)
      else:
        # Multi-class classification
        return torch.softmax(logits, dim=1)
