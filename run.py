# Import necessary libraries
import numpy as np
import pandas as pd
import pyproj
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize  # Import minimize function
import geopandas as gpd
from shapely.geometry import Point
import lightgbm as lgb
import seaborn as sns
import streamlit as st

# Define function for geometric median
def geometric_median(points: list):
    points = np.array(points)  # Convert to numpy array for array operations
    initial_guess = np.mean(points, axis=0)
    def objective_function(center):
        return np.sum(np.linalg.norm(points - center, axis=1))
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    return result.x

# Load the dataset
def load_data():
    dataset = pd.read_csv('modified_dataset.csv')
    return dataset

# Feature engineering
def feature_engineering(dataset):
    # Convert 'OBSERVATION_DATE' to datetime
    dataset['OBSERVATION_DATE'] = pd.to_datetime(dataset['OBSERVATION_DATE'])
    # Extract day of year and hour
    dataset['DAY_OF_YEAR'] = dataset['OBSERVATION_DATE'].dt.dayofyear
    dataset['HOUR'] = dataset['OBSERVATION_DATE'].dt.hour
    # Calculate geometric median for coordinates
    coords = list(zip(dataset['LATITUDE'], dataset['LONGITUDE']))
    center = geometric_median(coords)
    dataset['dist_bearing'] = dataset.apply(lambda row: get_dist_bearing(center, (row['LATITUDE'], row['LONGITUDE'])), axis=1)
    dataset['DIST'], dataset['DIREC'] = zip(*dataset['dist_bearing'])
    return dataset

# Data Cleaning
def data_cleaning(dataset):
    dataset = dataset.drop_duplicates()
    return dataset

# Define function for distance and bearing calculation
def get_dist_bearing(center, point):
    geodisc = pyproj.Geod(ellps='WGS84')
    lon1, lat1 = center
    lon2, lat2 = point
    fwd_azimuth, back_azimuth, distance = geodisc.inv(lon1, lat1, lon2, lat2)
    return distance, fwd_azimuth

# Main function to run the app
def main():
    st.title("Climate Data Analysis App")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Select an option", ["Overview", "Data Analysis"])

    if menu == "Overview":
        st.write("Welcome to the Climate Data Analysis App!")
        st.write("Geospatial analysis involves examining data that has a geographical or spatial component. It encompasses a wide range of techniques for analyzing and visualizing spatial data to derive meaningful insights.")
        st.write("Geospatial analysis finds applications in various fields, including environmental science, urban planning, transportation, epidemiology, and more.")
        st.write("In this app, we'll explore climate data using geospatial analysis techniques to gain insights into various climatic parameters such as temperature, elevation, and humidity.")
        st.write("Choose the 'Data Analysis' option from the sidebar to delve into the data and uncover interesting patterns.")

        # Load dataset
        dataset = load_data()

        # Feature Engineering
        dataset = feature_engineering(dataset)

        # Data Cleaning
        dataset = data_cleaning(dataset)

        # Define features
        features = [
            'DIST',  # Geographic coordinate: Latitude
            'DIREC',  # Geographic coordinate: Longitude
            'ELEVATION',  # Elevation above sea level
            'DAY_OF_YEAR',  # Day of the year (from OBSERVATION_DATE)
            'HOUR',  # Hour of the day (from OBSERVATION_DATE)
            'AIR_TEMPERATURE_DEW_POINT'  # Dew point temperature
        ]

        # Plot feature importances
        st.write("Exploring Feature Importance:")
        gbm = lgb.LGBMRegressor()
        gbm.fit(dataset[features], dataset['AIR_TEMPERATURE'])
        lgb.plot_importance(gbm, figsize=(10, 6))
        plt.title("Feature Importances")
        st.pyplot()

        # Explain feature importance
        st.write("The feature importance plot above displays the importance of various features in predicting air temperature.")
        st.write("From the plot, we observe that the most important features are day of the year, hour of the day, and dew point temperature.")
        st.write("Day of the year and hour of the day represent temporal patterns, capturing seasonal and diurnal variations in temperature.")
        st.write("Dew point temperature is crucial as it indicates the moisture content in the air, influencing how the temperature feels.")
        st.write("Understanding the importance of these features helps us better comprehend the factors affecting air temperature and aids in predictive modeling.")

    elif menu == "Data Analysis":
        st.subheader("Data Analysis Section")
        st.write("Performing data analysis...")

        # Load dataset
        dataset = load_data()

        # Feature Engineering
        dataset = feature_engineering(dataset)

        # Data Cleaning
        dataset = data_cleaning(dataset)

        # Define features
        features = [
            'DIST',  # Geographic coordinate: Latitude
            'DIREC',  # Geographic coordinate: Longitude
            'ELEVATION',  # Elevation above sea level
            'DAY_OF_YEAR',  # Day of the year (from OBSERVATION_DATE)
            'HOUR',  # Hour of the day (from OBSERVATION_DATE)
            'AIR_TEMPERATURE_DEW_POINT'  # Dew point temperature
        ]

        # Perform modeling and evaluation
        st.write("To perform geospatial analysis, we will conduct regression and distance metrics models.")
        st.write("For regression, we'll use Support Vector Regression (SVR), Random Forest, LightGBM, and K-Nearest Neighbors (KNN) models.")
        st.write("To evaluate these models, we'll use Mean Squared Error (MSE), which measures the average squared difference between the actual and predicted values.")
        st.write("After evaluating the models, we obtained the following results:")

        models = ['SVR', 'Random Forest', 'LightGBM', 'KNN']
        results = {}
        for model_name in models:
            # Train-test split
            X = dataset[features]
            y = dataset['AIR_TEMPERATURE']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            if model_name == 'SVR':
                model = SVR()
            elif model_name == 'Random Forest':
                model = RandomForestRegressor()
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor()
            elif model_name == 'KNN':
                model = KNeighborsRegressor()
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            results[model_name] = mse

        # Display results
        st.write("Model Evaluation Results:")
        for model_name, mse in results.items():
            st.write(f"- {model_name}: Mean Squared Error (MSE) = {mse}")

        # Add a map widget
        st.write("Displaying interactive map...")
        st.map(dataset[['LATITUDE', 'LONGITUDE']])

if __name__ == "__main__":
    # Specify the host and port for running the Streamlit app
    host = '0.0.0.0'  # Listen on all network interfaces
    port = 8501  # Use port 8501

    # Run the Streamlit app with the specified host and port
    main.run(host=host, port=port)

