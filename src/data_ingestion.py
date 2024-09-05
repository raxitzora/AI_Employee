import pandas as pd

def read_file(file_path, file_type):
    try:
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        elif file_type == 'excel':
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please choose from 'csv', 'json', or 'excel'.")
        return data
    
    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
        return None
    
    except ValueError as ve:
        print(ve)
        return None
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
