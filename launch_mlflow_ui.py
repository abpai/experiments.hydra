#!/usr/bin/env python3
"""
Launch MLflow UI to view experiment results.

Usage:
    python launch_mlflow_ui.py [--port PORT]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def main():
  """Launch MLflow UI."""
  parser = argparse.ArgumentParser(description='Launch MLflow UI')
  parser.add_argument('--port', type=int, default=5000, help='Port to run MLflow UI on')

  args = parser.parse_args()

  # Check if mlflow is installed
  try:
    import mlflow

    logger.info(f'MLflow version: {mlflow.__version__}')
  except ImportError:
    logger.error('❌ MLflow is not installed. Please install it with: uv add mlflow')
    sys.exit(1)

  # Check if mlruns directory exists
  mlruns_dir = Path('mlruns')
  if not mlruns_dir.exists():
    logger.warning('⚠️ No mlruns directory found. Run some experiments first.')
    logger.info('Try running: python train_scikit.py dataset=dummy_spam')

  # Launch MLflow UI
  logger.info(f'🚀 Launching MLflow UI on port {args.port}')
  logger.info(f'📊 View experiments at: http://localhost:{args.port}')
  logger.info('Press Ctrl+C to stop the UI')

  try:
    subprocess.run(
      [
        sys.executable,
        '-m',
        'mlflow',
        'ui',
        '--port',
        str(args.port),
        '--host',
        '0.0.0.0',
      ]
    )
  except KeyboardInterrupt:
    logger.info('👋 MLflow UI stopped')
  except Exception as e:
    logger.error(f'❌ Failed to start MLflow UI: {e}')
    sys.exit(1)


if __name__ == '__main__':
  main()
