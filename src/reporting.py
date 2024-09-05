import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data_distribution(data, columns=None):
     if columns is None:
        columns = data.select_dtypes(include=['float64', 'int64']).columns

     for column in columns:
        plt.figure(figsize=(14, 6))
        
        
        plt.subplot(1, 2, 1)
        sns.histplot(data[column], kde=True)
        plt.title(f'Histogram of {column}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[column])
        plt.title(f'Box Plot of {column}')
        
        plt.show()
        
def visualize_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
def visualize_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), [importances[i] for i in indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature importances.")
        
        
def visualize_clusters(data, model):
     if hasattr(model, 'labels_'):
        data['Cluster'] = model.labels_
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='Cluster', palette='viridis')
        plt.title('K-Means Clustering Visualization')
        plt.show()
     else:
        print("Model is not a KMeans clustering model.")
        
def generate_report(data, models, feature_columns):
    visualize_data_distribution(data, columns=feature_columns)
    visualize_correlation_matrix(data)
    
    for model_name, model in models.items():
        if model_name in ['DecisionTree', 'RandomForest']:
            visualize_feature_importance(model, feature_columns)
        elif model_name == 'KMeans':
            visualize_clusters(data, model)