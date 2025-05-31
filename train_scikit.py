import time

import hydra
import pandas as pd
import structlog
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from experiment_tracker import ExperimentTracker
from load_dataset import load_dataset

logger = structlog.get_logger(__name__)


@hydra.main(config_path='conf', config_name='config_scikit', version_base=None)
def train_model_pipeline(cfg: DictConfig) -> None:
  # Initialize experiment tracker
  tracker = ExperimentTracker()
  start_time = time.time()

  # Print the entire configuration object as soon as the function starts
  logger.info('--- ENTIRE HYDRA CONFIGURATION ---')
  logger.info(OmegaConf.to_yaml(cfg))
  logger.info('----------------------------------')

  # --- 1. Load Data (Simplified) ---
  X_text, y_labels = load_dataset(cfg)

  X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y_labels,
    test_size=cfg.dataset.test_split_ratio,
    random_state=cfg.experiment.random_seed,
    stratify=y_labels,  # Good practice for classification
  )
  logger.info('--- Data Split ---')
  logger.info(f'Train size: {len(X_train_text)}, Test size: {len(X_test_text)}')

  # --- 2. Feature Extraction ---
  # We can access feature_extractor parameters directly from cfg
  # If feature_extractor had a _target_, we could instantiate it too.
  logger.info('--- Initializing Feature Extractor (TF-IDF) ---')
  logger.info(f'TF-IDF Params: {OmegaConf.to_container(cfg.feature_extractor.params)}')

  # Hydra will instantiate the feature_extractor, but we need
  # to convert the ngram_range to a tuple
  vectorizer_params = {**cfg.feature_extractor.params}
  vectorizer_params['ngram_range'] = tuple(vectorizer_params['ngram_range'])
  vectorizer = TfidfVectorizer(**vectorizer_params)

  X_train_vec = vectorizer.fit_transform(X_train_text)
  X_test_vec = vectorizer.transform(X_test_text)
  logger.info(f'Shape of training features: {X_train_vec.shape}')
  logger.info(f'Shape of testing features: {X_test_vec.shape}')

  # --- 3. Instantiate Model using Hydra ---
  # hydra.utils.instantiate will create an instance of the model
  # specified by `_target_` in the model's config YAML,
  # passing other keys from that YAML as arguments to the model's constructor.
  logger.info(f'Model Class: {cfg.model._target_}')

  # Use Hydra's instantiate with the clean config and explicit target
  active_model = hydra.utils.instantiate(cfg.model)

  # --- 4. Train Model ---
  logger.info('--- Training Model ---')
  active_model.fit(X_train_vec, y_train)
  logger.info('Training complete.')

  # --- 5. Evaluate Model ---
  predictions = active_model.predict(X_test_vec)
  accuracy = accuracy_score(y_test, predictions)

  # Generate detailed classification report
  try:
    report = classification_report(
      y_test, predictions, output_dict=True, zero_division=0
    )
  except Exception as e:
    logger.error(f'Error generating classification report: {str(e)}')
    report = {}

  logger.info('--- Evaluation Results ---')
  logger.info(f'Selected Model: {cfg.model._target_}')
  logger.info(f'Accuracy on Test Set: {accuracy:.4f}')

  # Log experiment results
  training_time = time.time() - start_time
  metrics = {'accuracy': accuracy, 'report': report}
  tracker.log_experiment(
    cfg=cfg, metrics=metrics, model_type='scikit', training_time=training_time
  )

  # Hydra automatically saves the configuration for this run
  # in the outputs/ directory (or multirun/ for sweeps).
  run_dir = HydraConfig.get().runtime.output_dir
  logger.info(f'\nOutput and logs saved to: {run_dir}')
  logger.info('Experiment tracked in experiment_results/')


if __name__ == '__main__':
  train_model_pipeline()
