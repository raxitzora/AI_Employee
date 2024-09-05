import argparse
import pandas as pd
import pickle
import os
from data_ingestion import read_file
from preprocessing import data_preprocessing_pipeline
from analysis import analysis_engine
from reporting import generate_report

# Define a temporary file path for saving data between commands
TEMP_DATA_PATH = "temp_data.pkl"

def save_data(data):
    """ Save data to a pickle file """
    with open(TEMP_DATA_PATH, 'wb') as f:
        pickle.dump(data, f)

def read_file_from_file():
    """ Load data from a pickle file """
    if os.path.exists(TEMP_DATA_PATH):
        with open(TEMP_DATA_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="CLI for AI Project")
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    
    load_parser = subparsers.add_parser("read_file", help="Load data from a file")
    load_parser.add_argument("--file_path", type=str, required=True, help="Path to the data file (CSV, JSON, Excel)")
    load_parser.add_argument("--file_type", type=str, required=True, choices=["csv", "json", "excel"], help="Type of the data file (csv, json, excel)")
    
    preprocess_parser = subparsers.add_parser("preprocess_data", help="Preprocess the loaded data")

    analysis_parser = subparsers.add_parser("run_analysis", help="Run data analysis")
    analysis_parser.add_argument("--analysis_type", type=str, required=True, choices=["correlation", "regression", "decision_tree", "random_forest", "kmeans"], help="Type of analysis to perform")
    analysis_parser.add_argument("--target_variable", type=str, help="Target variable for regression or classification")
    analysis_parser.add_argument("--feature_variables", nargs='+', help="Feature variables for regression or classification")
    analysis_parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for K-Means")
    
    report_parser = subparsers.add_parser("generate_report", help="Generate report with visualizations")
    report_parser.add_argument("--feature_columns", nargs='+', help="Feature columns to include in the report")

    args = parser.parse_args()
    
    if args.command == "read_file":
        data = read_file(args.file_path, args.file_type)  # Load data
        save_data(data)  # Save the data to the temp file
        print("Data loaded and saved successfully.")
        
    elif args.command == "preprocess_data":
        data = read_file_from_file()  # Load the previously saved data
        if data is not None:
            data = data_preprocessing_pipeline(data)  # Preprocess the data
            save_data(data)  # Save the preprocessed data back
            print("Data preprocessed successfully.")
        else:
            print("No data loaded. Please load data first using the 'read_file' command.")

    elif args.command == "run_analysis":
        data = read_file_from_file()  # Load the previously saved data
        if data is not None:
            model = analysis_engine(data, args.analysis_type, args.target_variable, args.feature_variables, args.n_clusters)
            # Save the model (if needed) - adjust as per your project needs
            print(f"Analysis '{args.analysis_type}' completed successfully.")
        else:
            print("No data loaded. Please load data first using the 'read_file' command.")

    elif args.command == "generate_report":
        data = read_file_from_file()  # Load the data
        if data is not None:
            # Placeholder for loading the model (if needed)
            generate_report(data, None, args.feature_columns)
            print("Report generated successfully.")
        else:
            print("Data or model missing. Ensure data is loaded and analysis is run before generating a report.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
