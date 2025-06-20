#!/usr/bin/env python
"""
Main module for clustering analysis and sensitivity evaluation.

This script orchestrates the clustering of energy data, trains models on each cluster,
and conducts sensitivity analysis for different scenarios.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from utils.analysis_clustering import generate_dataset_cluster
from utils.models import run_model


def parse_arguments(custom_args=None):
    """Parse command line arguments.
    
    Args:
        custom_args: Optional list of arguments to parse. If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(description='Run clustering and sensitivity analysis of buildings')
    
    # Clustering parameters
    parser.add_argument('--data_path', type=str, default="data/clustering.csv",
                      help='Path to input CSV data file')
    parser.add_argument('--columns_selected', nargs='+', default=['QHnd', 'degree_days'],
                      help='Columns to use for clustering')
    parser.add_argument('--cluster_method_custom', action='store_true', default=False,
                      help='Use custom cluster method instead of statistical method')
    parser.add_argument('--cluster_value', type=int, default=2,
                      help='Number of clusters when using custom method')
    parser.add_argument('--cluster_method_stat', type=str, default="elbow",
                      help='Statistical method for determining cluster count (elbow, silhouette, etc.)')
    parser.add_argument('--columns_to_delete', nargs='+', 
                      default=["QHnd","EPl", "EPt", "EPc", "EPv", "EPw", "EPh", "QHimp", "theoric_nominal_power"],
                      help='Columns to remove from analysis')
    parser.add_argument('--save_clusters', action='store_true', default=True,
                      help='Whether to save cluster datasets')
    parser.add_argument('--clusters_output_dir', type=str, default="data/data_cluster",
                      help='Directory to save cluster datasets')
    
    # Modeling parameters
    parser.add_argument('--target', type=str, default="QHnd",
                      help='Target variable for prediction')
    parser.add_argument('--problem_type', type=str, default="regressione",
                      help='Type of problem (regressione/classificazione)')
    parser.add_argument('--models_dir', type=str, default="models",
                      help='Directory to save trained models')
    parser.add_argument('--sensitivity_vars', nargs='+', 
                      default=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'],
                      help='Variables to use for sensitivity analysis')
    parser.add_argument('--compare_scenarios', action='store_true', default=True,
                      help='Whether to compare different scenarios')
    parser.add_argument('--results_dir', type=str, default="result_sensitivity_cluster",
                      help='Directory to save sensitivity analysis results')
    parser.add_argument('--cluster_to_analyze', type=str, default=None,
                      help='Specific cluster to analyze (default: analyze all clusters)')
    
    if custom_args:
        return parser.parse_args(custom_args)
    else:
        return parser.parse_args()



def ensure_directories_exist(dirs):
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def main(custom_args=None, list_dict_scenarios=None):
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments(custom_args)
    
    # Create necessary directories
    ensure_directories_exist([args.clusters_output_dir, args.models_dir, args.results_dir])
    
    # Define scenarios for sensitivity analysis
    scenarios = list_dict_scenarios
    
    print("Loading dataset...")
    df = pd.read_csv(args.data_path, sep=",", decimal=".", low_memory=False, header=0, index_col=0)
    
    # Step 1: Generate clusters
    print(f"Generating clusters using {args.cluster_method_stat if not args.cluster_method_custom else 'custom'} method...")
    df_cluster, optimal_k = generate_dataset_cluster(
        df_=df,
        columns_selected=args.columns_selected,
        save_df_cluster=args.save_clusters,
        delete_columns=True,
        column_to_delete=args.columns_to_delete,
        cluster_method_custom=args.cluster_method_custom,
        cluster_value=args.cluster_value,
        cluster_method_stat=args.cluster_method_stat,
        path_folder_save_df_cluster=args.clusters_output_dir
    )
    
    print(f"Created {optimal_k} clusters.")
    
    # Step 2: Analyze clusters
    if args.cluster_to_analyze:
        # Analyze specific cluster
        cluster_list = [args.cluster_to_analyze]
    else:
        # Analyze all clusters
        cluster_list = [f"cluster_{i}" for i in range(optimal_k)]
    
    results = {}
    
    for cluster_name in tqdm(cluster_list, desc="Analyzing clusters"):
        print(f"\nProcessing {cluster_name}...")
        file_path_cluster = f"{args.clusters_output_dir}/{cluster_name}.csv"
        file_path_save_model = f'{args.models_dir}/{cluster_name}'
        
        try:
            predizioni_df, results_analisi_sensibilita, df_confronto_scenari = run_model(
                file_path_cluster=file_path_cluster,
                target_=args.target,
                problem_type_=args.problem_type,
                variables_for_sensitivity_analysis=args.sensitivity_vars,
                file_path_save_model=file_path_save_model,
                confronta_scenari_cluster=args.compare_scenarios,
                scenari=scenarios,
                path_save_result=f"{args.results_dir}/{cluster_name}",
                cluster_name=cluster_name
            )
            
            results[cluster_name] = {
                "predictions": predizioni_df,
                "sensitivity_results": results_analisi_sensibilita,
                "scenario_comparison": df_confronto_scenari
            }
            
            print(f"Successfully analyzed {cluster_name}")
        except Exception as e:
            print(f"Error analyzing {cluster_name}: {str(e)}")
    
    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    # custom_args = [
    #     '--cluster_value', '5',
    #     '--columns_selected', 'EPgl', 'degree_days',
    #     '--target', 'EPgl',
    #     '--problem_type', 'regression',
    # ]
    # list_dict_scenarios = [
    #     {'name': 'Scenario 1', 'parameters': {'average_opaque_surface_transmittance': 0.5, 
    #                                         'average_glazed_surface_transmittance': 1}},
    #     {'name': 'Scenario 2', 'parameters': {'average_opaque_surface_transmittance': 0.2, 
    #                                         'average_glazed_surface_transmittance': 0.7}}
    # ]
    main(custom_args=None, list_dict_scenarios=None)