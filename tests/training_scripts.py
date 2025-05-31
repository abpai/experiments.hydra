#!/usr/bin/env python3
"""
Test script to validate all training pipelines work
correctly with simplified configurations.

Usage:
    python tests/training_scripts.py
"""

import subprocess
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def run_command(cmd: str, description: str) -> bool:
  """Run a command and return True if successful."""
  try:
    logger.info(f'Testing: {description}')
    logger.info(f'Command: {cmd}')

    result = subprocess.run(
      cmd.split(),
      capture_output=True,
      text=True,
      timeout=300,  # 5 minute timeout
    )

    if result.returncode == 0:
      logger.info(f'✅ SUCCESS: {description}')
      return True
    else:
      logger.error(f'❌ FAILED: {description}')
      logger.error(f'STDOUT: {result.stdout}')
      logger.error(f'STDERR: {result.stderr}')
      return False

  except subprocess.TimeoutExpired:
    logger.error(f'❌ TIMEOUT: {description}')
    return False
  except Exception as e:
    logger.error(f'❌ ERROR: {description} - {str(e)}')
    return False


def test_configurations():
  """Test different configurations work correctly."""
  tests = [
    # Test scikit-learn with different configurations
    {
      'cmd': 'python train_scikit.py dataset=dummy_spam',
      'description': 'Scikit-learn with dummy dataset (default TF-IDF)',
    },
    {
      'cmd': 'python train_scikit.py dataset=dummy_spam feature_extractor=tfidf_advanced',
      'description': 'Scikit-learn with advanced TF-IDF',
    },
    {
      'cmd': 'python train_scikit.py dataset=dummy_spam feature_extractor=count_vectorizer',
      'description': 'Scikit-learn with Count Vectorizer',
    },
    {
      'cmd': 'python train_scikit.py dataset=dummy_spam model=logistic_regression',
      'description': 'Scikit-learn with Logistic Regression',
    },
    # Test neural network (disable validation split for small dummy dataset)
    {
      'cmd': 'python train_nn.py dataset=dummy_spam model.num_epochs=1 dataset.validation_split_ratio=0.0',
      'description': 'Neural Network with 1 epoch',
    },
    # Test transformer (disable validation split for small dummy dataset)
    {
      'cmd': 'python train_llm.py dataset=dummy_spam model.num_epochs=1 dataset.validation_split_ratio=0.0',
      'description': 'Transformer with 1 epoch',
    },
  ]

  results = []
  for test in tests:
    success = run_command(test['cmd'], test['description'])
    results.append((test['description'], success))

  return results


def main():
  """Run all tests and report results."""
  logger.info('🚀 Starting training script validation tests...')

  # Check if we're in the right directory
  if not Path('train_scikit.py').exists():
    logger.error('❌ Please run this script from the project root directory')
    sys.exit(1)

  # Check if MLflow is installed
  try:
    import mlflow

    logger.info('✅ MLflow is available for experiment tracking')
  except ImportError:
    logger.warning('⚠️ MLflow not installed - only custom tracking will work')

  # Run tests
  results = test_configurations()

  # Report summary
  logger.info('\n' + '=' * 60)
  logger.info('TEST SUMMARY')
  logger.info('=' * 60)

  passed = 0
  failed = 0

  for description, success in results:
    status = '✅ PASS' if success else '❌ FAIL'
    logger.info(f'{status}: {description}')
    if success:
      passed += 1
    else:
      failed += 1

  logger.info(f'\nTotal: {len(results)} tests')
  logger.info(f'Passed: {passed}')
  logger.info(f'Failed: {failed}')

  if failed == 0:
    logger.info('🎉 All tests passed!')
    sys.exit(0)
  else:
    logger.error(f'💥 {failed} tests failed!')
    sys.exit(1)


if __name__ == '__main__':
  main()
