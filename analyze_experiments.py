#!/usr/bin/env python3
"""
Experiment Results Analysis Tool

Usage:
    python analyze_experiments.py --top-n 10
    python analyze_experiments.py --model-type scikit
    python analyze_experiments.py --compare
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import structlog

from experiment_tracker import ExperimentTracker

logger = structlog.get_logger(__name__)


def setup_plotting():
  """Setup matplotlib and seaborn for better plots."""
  plt.style.use('default')
  sns.set_palette('husl')
  plt.rcParams['figure.figsize'] = (12, 8)


def plot_model_comparison(df: pd.DataFrame, save_path: Path = None):
  """Create comparison plots for different model types."""
  if len(df) == 0:
    logger.warning('No data to plot')
    return

  setup_plotting()

  # Create subplots
  fig, axes = plt.subplots(2, 2, figsize=(15, 12))
  fig.suptitle('Model Performance Comparison', fontsize=16)

  # 1. Accuracy by model type
  ax1 = axes[0, 0]
  df.boxplot(column='test_accuracy', by='model_type', ax=ax1)
  ax1.set_title('Test Accuracy by Model Type')
  ax1.set_xlabel('Model Type')
  ax1.set_ylabel('Test Accuracy')

  # 2. Accuracy over time
  ax2 = axes[0, 1]
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  for model_type in df['model_type'].unique():
    model_data = df[df['model_type'] == model_type]
    ax2.plot(
      model_data['timestamp'],
      model_data['test_accuracy'],
      marker='o',
      label=model_type,
      alpha=0.7,
    )
  ax2.set_title('Accuracy Over Time')
  ax2.set_xlabel('Timestamp')
  ax2.set_ylabel('Test Accuracy')
  ax2.legend()
  ax2.tick_params(axis='x', rotation=45)

  # 3. F1 Score distribution
  ax3 = axes[1, 0]
  model_types = df['model_type'].unique()
  for i, model_type in enumerate(model_types):
    model_data = df[df['model_type'] == model_type]['f1_macro']
    ax3.hist(model_data, alpha=0.6, label=model_type, bins=10)
  ax3.set_title('F1 Score Distribution')
  ax3.set_xlabel('F1 Score (Macro)')
  ax3.set_ylabel('Count')
  ax3.legend()

  # 4. Top performing models
  ax4 = axes[1, 1]
  top_10 = df.nlargest(10, 'test_accuracy')
  bars = ax4.bar(range(len(top_10)), top_10['test_accuracy'])
  ax4.set_title('Top 10 Models by Accuracy')
  ax4.set_xlabel('Model Rank')
  ax4.set_ylabel('Test Accuracy')

  # Color bars by model type
  colors = plt.cm.Set1(range(len(top_10['model_type'].unique())))
  color_map = dict(zip(top_10['model_type'].unique(), colors))
  for bar, model_type in zip(bars, top_10['model_type']):
    bar.set_color(color_map[model_type])

  # Add legend for model types
  legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=color_map[mt], label=mt)
    for mt in color_map.keys()
  ]
  ax4.legend(handles=legend_elements)

  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f'Plot saved to {save_path}')
  else:
    plt.show()


def print_summary_table(df: pd.DataFrame):
  """Print a formatted summary table."""
  if len(df) == 0:
    logger.warning('No data to summarize')
    return

  print('\n' + '=' * 80)
  print('EXPERIMENT RESULTS SUMMARY')
  print('=' * 80)

  # Overall stats
  print(f'Total experiments: {len(df)}')
  print(f'Model types: {", ".join(df["model_type"].unique())}')
  print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')

  print('\n' + '-' * 50)
  print('PERFORMANCE BY MODEL TYPE')
  print('-' * 50)

  summary = (
    df.groupby('model_type')
    .agg(
      {
        'test_accuracy': ['count', 'mean', 'std', 'max'],
        'f1_macro': ['mean', 'max'],
        'precision_macro': ['mean', 'max'],
        'recall_macro': ['mean', 'max'],
      }
    )
    .round(4)
  )

  print(summary)

  print('\n' + '-' * 50)
  print('TOP 5 PERFORMING MODELS')
  print('-' * 50)

  top_5 = df.nlargest(5, 'test_accuracy')[
    ['timestamp', 'model_type', 'model_class', 'test_accuracy', 'f1_macro']
  ]

  print(top_5.to_string(index=False))


def print_model_type_details(df: pd.DataFrame, model_type: str):
  """Print detailed results for a specific model type."""
  model_data = df[df['model_type'] == model_type]

  if len(model_data) == 0:
    logger.warning(f'No data found for model type: {model_type}')
    return

  print('\n' + '=' * 60)
  print(f'DETAILED RESULTS: {model_type.upper()}')
  print('=' * 60)

  # Best result
  best = model_data.loc[model_data['test_accuracy'].idxmax()]
  print(f'Best accuracy: {best["test_accuracy"]:.4f}')
  print(f'Best model: {best["model_class"]}')
  print(f'Timestamp: {best["timestamp"]}')

  # All results sorted by accuracy
  print(f'\nAll {model_type} results (sorted by accuracy):')
  print('-' * 40)

  columns = ['timestamp', 'model_class', 'test_accuracy', 'f1_macro']
  if model_type == 'scikit':
    columns.extend(['C', 'kernel', 'min_df', 'max_df'])
  elif model_type == 'neural_network':
    columns.extend(['hidden_units', 'learning_rate', 'batch_size'])
  elif model_type == 'transformer':
    columns.extend(['transformer_model', 'learning_rate', 'freeze_transformer'])

  available_columns = [col for col in columns if col in model_data.columns]
  sorted_results = model_data.sort_values('test_accuracy', ascending=False)[
    available_columns
  ]

  print(sorted_results.to_string(index=False))


def main():
  parser = argparse.ArgumentParser(description='Analyze experiment results')
  parser.add_argument(
    '--top-n', type=int, default=10, help='Number of top results to show'
  )
  parser.add_argument(
    '--model-type',
    choices=['scikit', 'neural_network', 'transformer'],
    help='Focus on specific model type',
  )
  parser.add_argument('--compare', action='store_true', help='Show comparison plots')
  parser.add_argument(
    '--save-plots', type=str, help='Save plots to file instead of showing'
  )

  args = parser.parse_args()

  # Initialize tracker and load data
  tracker = ExperimentTracker()

  if args.model_type:
    df = tracker.get_best_results(model_type=args.model_type)
    print_model_type_details(df, args.model_type)
  else:
    df = tracker.get_best_results()
    print_summary_table(df)

    if args.compare and len(df) > 0:
      save_path = Path(args.save_plots) if args.save_plots else None
      plot_model_comparison(df, save_path)


if __name__ == '__main__':
  main()
