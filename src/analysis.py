import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_analysis(data, target_variable):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    return correlation_matrix

def regression_analysis(data, target_variable, feature_variables):
    X = data[feature_variables]
    y = data[target_variable]
    model = LinearRegression()
    model.fit(X, y)
    print(f"Regression Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R^2 Score: {model.score(X, y)}")
    
    return model

def decision_tree_analysis(data, target_variable, feature_variables):
    X = data[feature_variables]
    y = data[target_variable]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
    
    return model

def random_forest_analysis(data, target_variable, feature_variables):
    X = data[feature_variables]
    y = data[target_variable]
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
    
    return model

def kmeans_clustering(data, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)
    model.fit(data)
    print(f"Cluster Centers: {model.cluster_centers_}")
    print(f"Inertia: {model.inertia_}")
    
    return model

def analysis_engine(data, analysis_type, target_variable=None, feature_variables=None, n_clusters=3):
    if analysis_type == 'correlation':
        return correlation_analysis(data, target_variable)
    elif analysis_type == 'regression':
        return regression_analysis(data, target_variable, feature_variables)
    elif analysis_type == 'decision_tree':
        return decision_tree_analysis(data, target_variable, feature_variables)
    elif analysis_type == 'random_forest':
        return random_forest_analysis(data, target_variable, feature_variables)
    elif analysis_type == 'kmeans':
        return kmeans_clustering(data, n_clusters)
    else:
        raise ValueError("Unsupported analysis type. Please choose from 'correlation', 'regression', 'decision_tree', 'random_forest', or 'kmeans'.")