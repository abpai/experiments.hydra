#!/usr/bin/env python3
"""
Experiment Results Analysis Tool

Analyzes MLflow experiment results across all experiments or specific ones.
By default, searches all experiments. Use --experiment to focus on one.

Usage:
    # Analyze all experiments
    python analyze_experiments.py

    # Focus on specific experiment
    python analyze_experiments.py --experiment spam_classification_experiment

    # Filter by model type
    python analyze_experiments.py --model-type scikit

    # Generate comparison plots
    python analyze_experiments.py --compare
    python analyze_experiments.py --compare --save-plots comparison.png

    # Combined usage
    python analyze_experiments.py --experiment nn_spam_classification --compare

Available experiments:
    - spam_classification_experiment (scikit-learn models)
    - nn_spam_classification (neural network models)
    - transformer_spam_classification (transformer models)
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import structlog

logger = structlog.get_logger(__name__)


def setup_plotting():
  """Setup matplotlib and seaborn for better plots."""
  plt.style.use('default')
  sns.set_palette('husl')
  plt.rcParams['figure.figsize'] = (12, 8)


def load_experiments_from_mlflow(
  model_type: Optional[str] = None, experiment_name: Optional[str] = None
) -> pd.DataFrame:
  """Load experiment data from MLflow tracking."""
  try:
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri('file:./mlruns')

    # Get experiments to search
    if experiment_name:
      # Search specific experiment
      experiment = mlflow.get_experiment_by_name(experiment_name)
      if experiment is None:
        logger.warning(f"No MLflow experiment found named '{experiment_name}'")
        available_experiments = [exp.name for exp in mlflow.search_experiments()]
        logger.info(f'Available experiments: {available_experiments}')
        return pd.DataFrame()
      experiment_ids = [experiment.experiment_id]
      logger.info(f'Analyzing experiment: {experiment_name}')
    else:
      # Search all experiments
      experiments = mlflow.search_experiments()
      experiment_ids = [exp.experiment_id for exp in experiments]
      logger.info(f'Analyzing all experiments: {[exp.name for exp in experiments]}')

    # Search for all runs across selected experiments
    # Use search_runs without experiment filter to get all data
    try:
      runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        output_format='pandas',
        search_all_experiments=False,
        max_results=1000,
      )
    except Exception as e:
      logger.warning(f'Error with advanced search, trying simple search: {e}')
      # Fallback to simple search
      all_runs = []
      for exp_id in experiment_ids:
        try:
          exp_runs = mlflow.search_runs(experiment_ids=[exp_id])
          if len(exp_runs) > 0:
            all_runs.append(exp_runs)
        except Exception as e:
          logger.warning(f'Error searching experiment {exp_id}: {e}')

      if not all_runs:
        logger.warning('No MLflow runs found in any experiment')
        return pd.DataFrame()

      runs = pd.concat(all_runs, ignore_index=True)

    if len(runs) == 0:
      logger.warning('No MLflow runs found in any experiment')
      return pd.DataFrame()

    logger.info(f'Found {len(runs)} total runs across all experiments')

    # Create a simplified dataframe with the key columns
    df_data = []

    for _, run in runs.iterrows():
      # Extract basic info
      run_data = {
        'run_id': run['run_id'],
        'timestamp': pd.to_datetime(run['start_time']).strftime('%Y-%m-%d_%H-%M-%S'),
        'status': run['status'],
        'experiment_name': run.get('experiment_id', 'unknown'),  # Will map this below
      }

      # Extract metrics
      metrics_cols = [col for col in run.index if col.startswith('metrics.')]
      for col in metrics_cols:
        metric_name = col.replace('metrics.', '')
        run_data[metric_name] = run[col]

      # Extract parameters
      param_cols = [col for col in run.index if col.startswith('params.')]
      for col in param_cols:
        param_name = col.replace('params.', '')
        run_data[param_name] = run[col]

      # Extract tags for model type identification
      tag_cols = [col for col in run.index if col.startswith('tags.')]
      for col in tag_cols:
        tag_name = col.replace('tags.', '')
        run_data[tag_name] = run[col]

      df_data.append(run_data)

    df = pd.DataFrame(df_data)

    # Map experiment IDs to names
    exp_id_to_name = {
      exp.experiment_id: exp.name for exp in mlflow.search_experiments()
    }
    df['experiment_name'] = df['experiment_name'].map(exp_id_to_name)

    # Map model types to framework names for filtering
    model_type_mapping = {
      'scikit': 'scikit-learn',
      'neural_network': 'pytorch-embeddings',
      'transformer': 'pytorch-transformers',
    }

    # Filter by model type if specified
    if model_type:
      framework_name = model_type_mapping.get(model_type)
      if framework_name and 'framework' in df.columns:
        df = df[df['framework'] == framework_name]
        logger.info(
          f'Filtered to {len(df)} runs for model type: {model_type} '
          f'(framework: {framework_name})'
        )
      else:
        logger.warning(f'No framework mapping found for model type: {model_type}')

    # Only return successful runs
    df = df[df['status'] == 'FINISHED']
    logger.info(f'Found {len(df)} successful runs')

    # Ensure we have key columns, set defaults if missing
    key_columns = ['test_accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    for col in key_columns:
      if col not in df.columns:
        df[col] = 0.0

    return df

  except Exception as e:
    logger.error(f'Error loading experiments from MLflow: {str(e)}')
    return pd.DataFrame()


def plot_model_comparison(df: pd.DataFrame, save_path: Path = None):
  """Create comparison plots for different model types."""
  if len(df) == 0:
    logger.warning('No data to plot')
    return

  setup_plotting()

  # Create subplots
  fig, axes = plt.subplots(2, 2, figsize=(15, 12))
  fig.suptitle('Model Performance Comparison', fontsize=16)

  # 1. Accuracy by model type (if framework column exists)
  ax1 = axes[0, 0]
  if 'framework' in df.columns and df['framework'].nunique() > 1:
    # Filter out rows with NaN test_accuracy for boxplot
    valid_df = df.dropna(subset=['test_accuracy'])
    if len(valid_df) > 0:
      valid_df.boxplot(column='test_accuracy', by='framework', ax=ax1)
      ax1.set_title('Test Accuracy by Framework')
      ax1.set_xlabel('Framework')
      ax1.set_ylabel('Test Accuracy')
    else:
      ax1.text(
        0.5,
        0.5,
        'No valid accuracy data',
        ha='center',
        va='center',
        transform=ax1.transAxes,
      )
  else:
    valid_accuracy = df['test_accuracy'].dropna()
    if len(valid_accuracy) > 0:
      ax1.hist(valid_accuracy, bins=10, alpha=0.7)
      ax1.set_title('Test Accuracy Distribution')
      ax1.set_xlabel('Test Accuracy')
      ax1.set_ylabel('Count')
    else:
      ax1.text(
        0.5,
        0.5,
        'No valid accuracy data',
        ha='center',
        va='center',
        transform=ax1.transAxes,
      )

  # 2. Accuracy over time
  ax2 = axes[0, 1]
  try:
    # Parse timestamp format: 2025-05-31_22-26-01
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d_%H-%M-%S')
  except (ValueError, TypeError):
    # Fallback to automatic parsing
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

  if 'framework' in df.columns and df['framework'].nunique() > 1:
    for framework in df['framework'].unique():
      framework_data = df[df['framework'] == framework].dropna(subset=['test_accuracy'])
      if len(framework_data) > 0:  # Only plot if we have valid data
        ax2.plot(
          framework_data['timestamp'],
          framework_data['test_accuracy'],
          marker='o',
          label=framework,
          alpha=0.7,
        )
    ax2.legend()
  else:
    valid_data = df.dropna(subset=['test_accuracy'])
    if len(valid_data) > 0:
      ax2.plot(
        valid_data['timestamp'], valid_data['test_accuracy'], marker='o', alpha=0.7
      )

  ax2.set_title('Accuracy Over Time')
  ax2.set_xlabel('Timestamp')
  ax2.set_ylabel('Test Accuracy')
  ax2.tick_params(axis='x', rotation=45)

  # 3. F1 Score distribution
  ax3 = axes[1, 0]
  if 'framework' in df.columns and df['framework'].nunique() > 1:
    frameworks = df['framework'].unique()
    for i, framework in enumerate(frameworks):
      framework_data = df[df['framework'] == framework]['f1_macro'].dropna()
      if len(framework_data) > 0:  # Only plot if we have valid data
        ax3.hist(framework_data, alpha=0.6, label=framework, bins=10)
    ax3.legend()
  else:
    valid_f1_data = df['f1_macro'].dropna()
    if len(valid_f1_data) > 0:
      ax3.hist(valid_f1_data, alpha=0.7, bins=10)

  ax3.set_title('F1 Score Distribution')
  ax3.set_xlabel('F1 Score (Macro)')
  ax3.set_ylabel('Count')

  # 4. Top performing models
  ax4 = axes[1, 1]
  top_10 = df.nlargest(10, 'test_accuracy')
  bars = ax4.bar(range(len(top_10)), top_10['test_accuracy'])
  ax4.set_title('Top 10 Models by Accuracy')
  ax4.set_xlabel('Model Rank')
  ax4.set_ylabel('Test Accuracy')

  # Color bars by framework if available
  if 'framework' in top_10.columns:
    colors = plt.cm.Set1(range(len(top_10['framework'].unique())))
    color_map = dict(zip(top_10['framework'].unique(), colors))
    for bar, framework in zip(bars, top_10['framework']):
      bar.set_color(color_map[framework])

    # Add legend for frameworks
    legend_elements = [
      plt.Rectangle((0, 0), 1, 1, color=color_map[fw], label=fw)
      for fw in color_map.keys()
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
  if 'experiment_name' in df.columns:
    exp_counts = df['experiment_name'].value_counts()
    print(f'Experiments: {dict(exp_counts)}')
  if 'framework' in df.columns:
    print(f'Frameworks: {", ".join(df["framework"].unique())}')
  print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')

  print('\n' + '-' * 50)
  print('PERFORMANCE BY FRAMEWORK')
  print('-' * 50)

  if 'framework' in df.columns and df['framework'].nunique() > 1:
    summary = (
      df.groupby('framework')
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
  else:
    print(f'Average accuracy: {df["test_accuracy"].mean():.4f}')
    print(f'Best accuracy: {df["test_accuracy"].max():.4f}')
    print(f'Standard deviation: {df["test_accuracy"].std():.4f}')

  print('\n' + '-' * 50)
  print('TOP 5 PERFORMING MODELS')
  print('-' * 50)

  display_cols = ['timestamp', 'test_accuracy', 'f1_macro']
  if 'framework' in df.columns:
    display_cols.insert(1, 'framework')
  if 'experiment_name' in df.columns:
    display_cols.insert(-1, 'experiment_name')

  available_cols = [col for col in display_cols if col in df.columns]
  top_5 = df.nlargest(5, 'test_accuracy')[available_cols]

  print(top_5.to_string(index=False))


def print_model_type_details(df: pd.DataFrame, model_type: str):
  """Print detailed results for a specific model type."""
  if len(df) == 0:
    logger.warning(f'No data found for model type: {model_type}')
    return

  print('\n' + '=' * 60)
  print(f'DETAILED RESULTS: {model_type.upper()}')
  print('=' * 60)

  # Check if we have valid accuracy data
  valid_accuracy_df = df.dropna(subset=['test_accuracy'])

  if len(valid_accuracy_df) == 0:
    print('No runs with valid test accuracy found.')
    print(f'Total runs: {len(df)}')
    return

  # Best result
  best_idx = valid_accuracy_df['test_accuracy'].idxmax()
  best = valid_accuracy_df.loc[best_idx]
  print(f'Best accuracy: {best["test_accuracy"]:.4f}')
  print(f'Timestamp: {best["timestamp"]}')
  print(f'Run ID: {best["run_id"]}')

  # All results sorted by accuracy
  print(f'\nAll {model_type} results (sorted by accuracy):')
  print('-' * 40)

  columns = ['timestamp', 'test_accuracy', 'f1_macro']

  # Add model-specific columns if they exist
  potential_cols = [
    'C',
    'kernel',
    'min_df',
    'max_df',
    'hidden_units',
    'learning_rate',
    'batch_size',
  ]
  for col in potential_cols:
    if col in df.columns:
      columns.append(col)

  available_columns = [col for col in columns if col in df.columns]
  sorted_results = df.sort_values('test_accuracy', ascending=False, na_position='last')[
    available_columns
  ]

  print(sorted_results.to_string(index=False))


def main():
  parser = argparse.ArgumentParser(description='Analyze experiment results from MLflow')
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
  parser.add_argument(
    '--experiment',
    type=str,
    help='Specific experiment name to analyze (default: all experiments)',
  )

  args = parser.parse_args()

  # Load data from MLflow
  if args.model_type:
    df = load_experiments_from_mlflow(
      model_type=args.model_type, experiment_name=args.experiment
    )
    print_model_type_details(df, args.model_type)
  else:
    df = load_experiments_from_mlflow(experiment_name=args.experiment)
    print_summary_table(df)

    if args.compare and len(df) > 0:
      save_path = Path(args.save_plots) if args.save_plots else None
      plot_model_comparison(df, save_path)


if __name__ == '__main__':
  main()
