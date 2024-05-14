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
from scipy.optimize import minimize
import geopandas as gpd
from shapely.geometry import Point
import lightgbm as lgb
import seaborn as sns
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

def geometric_median(points: list):
    points = np.array(points)
    initial_guess = np.mean(points, axis=0)
    def objective_function(center):
        return np.sum(np.linalg.norm(points - center, axis=1))
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    return result.x

def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def feature_engineering(dataset):
    dataset['OBSERVATION_DATE'] = pd.to_datetime(dataset['OBSERVATION_DATE'])
    dataset['DAY_OF_YEAR'] = dataset['OBSERVATION_DATE'].dt.dayofyear
    dataset['HOUR'] = dataset['OBSERVATION_DATE'].dt.hour
    coords = list(zip(dataset['LATITUDE'], dataset['LONGITUDE']))
    center = geometric_median(coords)
    dataset['dist_bearing'] = dataset.apply(lambda row: get_dist_bearing(center, (row['LATITUDE'], row['LONGITUDE'])), axis=1)
    dataset['DIST'], dataset['DIREC'] = zip(*dataset['dist_bearing'])
    return dataset

def data_cleaning(dataset):
    dataset = dataset.drop_duplicates()
    return dataset

def get_dist_bearing(center, point):
    geodisc = pyproj.Geod(ellps='WGS84')
    lon1, lat1 = center
    lon2, lat2 = point
    fwd_azimuth, back_azimuth, distance = geodisc.inv(lon1, lat1, lon2, lat2)
    return distance, fwd_azimuth

def main():
    st.title("Geospatial analysis")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Select an option", ["Overview", "Data Analysis"])

    if menu == "Overview":
        st.subheader("Overview")
        st.write("Geospatial analysis")
        st.write("Geospatial analysis is a powerful approach for examining and interpreting data that possesses a spatial or geographical component. By leveraging geographic information systems (GIS) and various analytical techniques, geospatial analysis allows us to uncover valuable insights about our world.")
        st.write("In this project, our focus is on exploring climate data within the context of the Kingdom of Saudi Arabia. By harnessing geospatial analysis techniques, we aim to gain a deeper understanding of the intricate relationships between climatic factors and geographical features within this region.")
        st.write("The Kingdom of Saudi Arabia, with its vast and diverse landscape, presents an intriguing dataset for geospatial analysis. Across different geographical locations offers a rich tapestry for exploration.")
        st.write("Through this app, we'll delve into climate data sourced from various observation stations across Saudi Arabia. By examining parameters such as temperature, elevation, and humidity, we'll unravel spatial patterns and correlations, shedding light on the complex interplay between environmental factors.")
        st.write("Join us on this journey as we employ geospatial analysis techniques to decipher the climate data of the Kingdom of Saudi Arabia, uncovering insights that may help scientific research.")

    elif menu == "Data Analysis":
        st.subheader("Data Analysis Section")
        st.write("Welcome to the Data Analysis section! In this section, we will explore various geospatial analysis techniques to gain insights into climate data related to the Kingdom of Saudi Arabia.")
        st.write("Geospatial analysis involves analyzing and visualizing data that has a geographical or spatial component. It helps us understand how different climatic parameters vary across space and time.")
        st.write("In this project, we are particularly interested in studying climate data, including temperature, elevation, and humidity, to uncover spatial patterns and correlations in the Kingdom of Saudi Arabia.")
        st.write("One of the key techniques we'll be using is kriging, a geostatistical method for interpolating spatial data. However, since kriging is not directly supported by Streamlit's mapping functionality, we'll focus on other regression models for analysis.")

        file_path = 'modified_dataset.csv'
        dataset = load_data(file_path)
        dataset = feature_engineering(dataset)
        dataset = data_cleaning(dataset)

        features = [
            'DIST',
            'DIREC',
            'ELEVATION',
            'DAY_OF_YEAR',
            'HOUR',
            'AIR_TEMPERATURE_DEW_POINT'
        ]

        st.write("Let's start by exploring regression models for predicting air temperature based on various features.")
        st.write("We'll evaluate the following regression models:")
        st.write("- Support Vector Regression (SVR)")
        st.write("- gradient boosting regressor")
        st.write("- LightGBM")
        st.write("- K-Nearest Neighbors (KNN)")
        st.write("- Ridge Regression")

        st.write("After training and testing each model, we'll compare their performance using R^2.")

        models = ['SVR', 'Random Forest', 'LightGBM', 'KNN']
        results = {}
        for model_name in models:
            X = dataset[features]
            y = dataset['AIR_TEMPERATURE']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_name == 'SVR':
                model = SVR()
            elif model_name == 'Random Forest':
                model = RandomForestRegressor()
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor()
            elif model_name == 'KNN':
                model = KNeighborsRegressor()
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            results[model_name] = mse

        # st.write("Here are the model evaluation results:")
        # for model_name, mse in results.items():
        #     mse_formatted = round(mse, 2)
        #     st.write(f"- {model_name}: Mean Squared Error (MSE) = {mse_formatted}")

        st.write("Based on the evaluation results, it's evident that the performance of the different regression models varies .")
        st.write("1.KNN had the highest ranking")
        st.write("2. LightGBM is second")
        st.write("3.the rest of the models are quite close to eachother in terms they all did worse than our top ranking models")

        # Integrate KNN model evaluation
        st.write("Let's also employ the K-Nearest Neighbors (KNN) model:")
        st.write("Please input the following features:")
        
        start_date = datetime.today() - timedelta(days=365*24)
        end_date = datetime.today()

        
    

        # Input fields
        dist_input = st.number_input("Distance", value=0.0)
        direc_input = st.number_input("Direction", value=0.0)
        elevation_input = st.number_input("Elevation", value=1, min_value=1, max_value=10)
        day_input = st.selectbox("Day of Year", [str(day) for day in range(1, 366)])
        hour_input = st.selectbox("Hour", [str(hour) for hour in range(1, 25)])
        dew_point_input = st.number_input("Dew Point", value=0.0)

        # Convert input values to numeric data types
        dist_input = float(dist_input)
        direc_input = float(direc_input)
        elevation_input = int(elevation_input)
        day_input = float(day_input)
        hour_input = float(hour_input)
        dew_point_input = float(dew_point_input)

        # KNN model prediction
        knn_features = [dist_input, direc_input, elevation_input, day_input, hour_input, dew_point_input]
        knn_X = np.array(knn_features).reshape(1, -1)

        # Convert knn_X to float64
        knn_X = knn_X.astype(np.float64)

        # KNN model prediction
        knn_model = KNeighborsRegressor()
        knn_model.fit(X_train, y_train)
        knn_prediction = knn_model.predict(knn_X)

        st.write(f"Predicted Air Temperature (KNN): {knn_prediction[0]:.2f} °C")


if __name__ == "__main__":
    main()
