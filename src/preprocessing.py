import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def handle_missing_values(data,strategy='mean',columns=None):
    if columns is None:
        columns=data.columns
        
    if strategy=="mean":
        data[columns] = data[columns].fillna(data[columns].mean())
    elif strategy == 'median':
        data[columns] = data[columns].fillna(data[columns].median())
    elif strategy == 'drop': 
        data = data.dropna(subset=columns)
    else:
        raise ValueError("Unsupported strategy. Please choose from 'mean', 'median', or 'drop'.")  
    return data    



def normalize_data(data, method='minmax', columns=None):
    if columns is None:
        columns = data.select_dtypes(include=['float64', 'int64']).columns
    if method == 'minmax':
        scaler = MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns])
    elif method == 'zscore':
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
    else:
        raise ValueError("Unsupported method. Please choose from 'minmax' or 'zscore'.")
    
    return data

def data_preprocessing_pipeline(data, missing_values_strategy='mean', normalization_method='minmax'):
    data = handle_missing_values(data, strategy=missing_values_strategy)
    data = normalize_data(data, method=normalization_method)
    
    return data