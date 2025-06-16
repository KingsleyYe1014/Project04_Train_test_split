import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from config import CENSUS_API_KEY

def fetch_census_data():
    """
    Fetch data from Census API
    Returns a pandas DataFrame with the data
    """
    # Base URL for Census API
    base_url = "https://api.census.gov/data/2021/acs/acs5"
    
    # Variables we want to get
    variables = [
        "B19013_001E",  # Median household income
        "B15003_022E",  # Bachelor's degree
        "B15003_023E",  # Master's degree
        "B15003_024E",  # Professional degree
        "B15003_025E",  # Doctorate degree
        "B23025_002E",  # In labor force
        "B23025_007E",  # Not in labor force
        "B01003_001E"   # Total population
    ]
    
    # Parameters for the API request
    params = {
        "get": ",".join(variables),
        "for": "state:*",
        "key": CENSUS_API_KEY
    }
    
    try:
        # Make the API request
        print("Making API request...")
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Print response status and content for debugging
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")  # Print first 200 chars
        
        # Parse JSON response
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Convert columns to numeric
        for col in df.columns:
            if col != "state":
                df[col] = pd.to_numeric(df[col])
        
        print(f"Successfully fetched data for {len(df)} states")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        raise
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response content: {response.text}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def prepare_data(df):
    """
    Prepare data for modeling
    """
    # Calculate education level (percentage of population with bachelor's or higher)
    df['education_level'] = (df['B15003_022E'] + df['B15003_023E'] + 
                           df['B15003_024E'] + df['B15003_025E']) / df['B01003_001E']
    
    # Calculate labor force participation rate
    df['labor_force_rate'] = df['B23025_002E'] / (df['B23025_002E'] + df['B23025_007E'])
    
    # Calculate per capita income
    df['per_capita_income'] = df['B19013_001E'] / df['B01003_001E']
    
    # Select features and target
    X = df[['education_level', 'labor_force_rate']]
    y = df['B19013_001E']
    
    return X, y, df

def analyze_correlations(df):
    """
    Analyze correlations between variables
    """
    # Calculate correlation matrix
    corr_matrix = df[['B19013_001E', 'education_level', 'labor_force_rate', 'per_capita_income']].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    return corr_matrix

def train_models(X, y):
    """
    Train and evaluate models
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, model.coef_))
        else:
            feature_importance = None
        
        results[name] = {
            'model': model,
            'r2_score': r2,
            'mse': mse,
            'predictions': y_pred,
            'actual': y_test,
            'feature_importance': feature_importance
        }
    
    return results

def plot_results(results, X):
    """
    Create visualizations of the results
    """
    # Plot predictions vs actual
    for name, result in results.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(result['actual'], result['predictions'], alpha=0.5)
        plt.plot([result['actual'].min(), result['actual'].max()], 
                [result['actual'].min(), result['actual'].max()], 
                'r--', lw=2)
        plt.xlabel('Actual Income')
        plt.ylabel('Predicted Income')
        plt.title(f'{name} Predictions vs Actual')
        plt.savefig(f'{name.lower().replace(" ", "_")}_predictions.png')
        plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        if result['feature_importance'] is not None:
            importance = list(result['feature_importance'].values())
            plt.bar(X.columns, importance, alpha=0.5, label=name)
    plt.title('Feature Importance Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Fetch data
    print("Fetching data from Census API...")
    df = fetch_census_data()
    
    # Prepare data
    print("Preparing data...")
    X, y, df = prepare_data(df)
    
    # Analyze correlations
    print("Analyzing correlations...")
    corr_matrix = analyze_correlations(df)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Train models
    print("\nTraining models...")
    results = train_models(X, y)
    
    # Print results
    print("\nModel Results:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"R² Score: {result['r2_score']:.4f}")
        print(f"MSE: {result['mse']:.2f}")
        if result['feature_importance'] is not None:
            print("Feature Importance:")
            for feature, importance in result['feature_importance'].items():
                print(f"  {feature}: {importance:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(results, X)

if __name__ == "__main__":
    main() 