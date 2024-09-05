# cli_interface.py

import argparse
import pandas as pd
from data_ingestion import read_file
from preprocessing import data_preprocessing_pipeline
from analysis import analysis_engine
from reporting import generate_report
from preprocessing import data_preprocessing_pipeline

def handle_nlp_input(user_query):
    """Process natural language input and trigger appropriate actions."""
    query_info = data_preprocessing_pipeline(user_query)

    if query_info["command"] == "read_file":
        if query_info["file_path"]:
            data = read_file(query_info["file_path"])
            print("Data loaded successfully.")
        else:
            print("Please provide a file path.")

    elif query_info["command"] == "preprocess_data":
        if 'data' in locals():
            data = data_preprocessing_pipeline(data)
            print("Data preprocessed successfully.")
        else:
            print("No data loaded. Please load data first.")

    elif query_info["command"] == "run_analysis":
        if 'data' in locals():
            if query_info["analysis_type"]:
                model = analysis_engine(data, query_info["analysis_type"], query_info["target_variable"], query_info["feature_columns"])
                print(f"Analysis '{query_info['analysis_type']}' completed successfully.")
            else:
                print("Please specify an analysis type.")
        else:
            print("No data loaded. Please load data first.")

    elif query_info["command"] == "generate_report":
        if 'data' in locals() and 'model' in locals():
            generate_report(data, model, query_info["feature_columns"])
            print("Report generated successfully.")
        else:
            print("Ensure data is loaded and analysis is run before generating a report.")

def main():
    parser = argparse.ArgumentParser(description="CLI for AI Project")

    # Option for natural language input
    parser.add_argument("--nlp", type=str, help="Enter your command in natural language.")

    # Parse arguments
    args = parser.parse_args()

    if args.nlp:
        handle_nlp_input(args.nlp)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
