import numpy as np
import pandas as pd
import pyproj
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
from shapely.geometry import Point
import lightgbm as lgb
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from scipy.optimize import minimize
import folium
from streamlit_folium import folium_static



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


def page_overview():
    st.title("Overview")
    st.write("Geospatial analysis")
    st.write("Geospatial analysis is a powerful approach for examining and interpreting data that possesses a spatial or geographical component. By leveraging geographic information systems (GIS) and various analytical techniques, geospatial analysis allows us to uncover valuable insights about our world.")
    st.write("In this project, our focus is on exploring climate data within the context of the Kingdom of Saudi Arabia. By harnessing geospatial analysis techniques, we aim to gain a deeper understanding of the intricate relationships between climatic factors and geographical features within this region.")
    st.write("The Kingdom of Saudi Arabia, with its vast and diverse landscape, presents an intriguing dataset for geospatial analysis. Across different geographical locations offers a rich tapestry for exploration.")
    st.write("Through this app, we'll delve into climate data sourced from various observation stations across Saudi Arabia. By examining parameters such as temperature, elevation, and humidity, we'll unravel spatial patterns and correlations, shedding light on the complex interplay between environmental factors.")
    st.write("Join us on this journey as we employ geospatial analysis techniques to decipher the climate data of the Kingdom of Saudi Arabia, uncovering insights that may help scientific research.")
    st.write("If your intrested in the main regions of our data we have pinpointed them on this map!")

    m = folium.Map(location=[24.774265, 46.738586], zoom_start=6)  # Coordinates for Riyadh, Saudi Arabia
    
    # Add markers for specific locations
    # Example markers (you can replace these with your actual data)
    marker1 = folium.Marker(location=[24.7136, 46.6753], popup="Marker 1: Riyadh")
    marker2 = folium.Marker(location=[21.4858, 39.1925], popup="Marker 2: Jeddah")
    
    # Add markers to the map
    m.add_child(marker1)
    m.add_child(marker2)
    
    # Render the map in Streamlit
    folium_static(m)
    
def page_data_analysis():
    st.title("Data Analysis")
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
    st.write("- Gradient Boosting Regressor")
    st.write("- LightGBM")
    st.write("- K-Nearest Neighbors (KNN)")

    st.write("After training and testing each model, we'll compare their performance using R^2 and Mean Squared Error.")
    

    models = ['SVR', 'gradiant boost', 'LightGBM', 'KNN']
    results = {}
    for model_name in models:
        X = dataset[features]
        y = dataset['AIR_TEMPERATURE']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_name == 'SVR':
            model = SVR()
        elif model_name == 'gradiant boost':
            model = GradientBoostingRegressor()
        elif model_name == 'LightGBM':
            model = lgb.LGBMRegressor()
        elif model_name == 'KNN':
            model = KNeighborsRegressor()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {'MSE': mse, 'R^2': r2}

    mse_r2_df = pd.DataFrame(results).T
    st.write("Model Evaluation:")
    st.bar_chart(mse_r2_df)

    st.write("Based on the evaluation results and visuals, it was evident that the performance of the different regression models varies . the top ranked models were:")
    st.write("1.KNN had the highest ranking")
    st.write("2. LightGBM is second")
    st.write("the rest of the models are quite close to eachother and preformed rather ineffciently. But gradiant boost stood out as the worst.")
    st.write("This shows that for real-time predication we can deploy either KNN or lightgbm as they were good at capturing the patterns of our data. The others not so much.")
    st.write("Also,a histogram showing the distribution of temperatures.The bars on the graph  represent the number of times a certain temperature range was measured. For example, the tallest bar appears to be around 10 degrees Celsius, which could mean that the temperature range around 10 degrees Celsius was the most common temperature that was measured.")
    fig, ax = plt.subplots()
    sns.histplot(dataset['AIR_TEMPERATURE'], bins=20, kde=True, ax=ax)
    ax.set_title("Temperature Distribution")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(1, 40)
    # Render the bar chart in Streamlit
    st.pyplot(fig)

    st.write("the graph suggests that the most common temperatures fall between 10 and 20 degrees Celsius.")
    st.write("What can we duduce from this? for example. \n Understanding growing seasons: The 10-20°C range suggests a moderate climate. This might indicate two or even four growing seasons depending on the extremes outside this range.")

     # Now let's visualize the predicted air temperatures with an area chart based on temperature ranges
    st.write("Below is an area chart showing the predicted air temperatures grouped by temperature ranges:")
    temperature_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
    temperature_counts = []

    for temp_range in temperature_ranges:
        count = dataset[(dataset['AIR_TEMPERATURE'] >= temp_range[0]) & (dataset['AIR_TEMPERATURE'] < temp_range[1])].shape[0]
        temperature_counts.append(count)

    temperature_df = pd.DataFrame({
        'Temperature Range': [f"{temp_range[0]}-{temp_range[1]} °C" for temp_range in temperature_ranges],
        'Count': temperature_counts
    })

    st.area_chart(temperature_df.set_index('Temperature Range'))
    st.write("Again it seems the most is within 10-20 degrees, so we can make a lot of , Agriculture,Climate and Clothing,Energy Consumption. And many more decisons based on this result for KSA.")
def page_now_you_try():
    st.title("Now You Try")
    st.write("Let's also employ the K-Nearest Neighbors (KNN) model:")
    st.write("Please input the following features:")

    start_date = datetime.today() - timedelta(days=365 * 24)
    end_date = datetime.today()

    # Input fields
    dist_input = st.number_input("Distance", value=0.0)
    direc_input = st.number_input("Direction", value=0.0)
    elevation_input = st.number_input("Elevation", value=1, min_value=1, max_value=10)
    day_input = st.selectbox("Day of Year", [str(day) for day in range(1, 366)])
    hour_input = st.selectbox("Hour", [str(hour) for hour in range(1, 25)])
    dew_point_input = st.number_input("Dew Point", value=0.0)

    # Convert input values to numeric data types
    day_input = float(day_input)
    hour_input = float(hour_input)

    # KNN model prediction
    knn_features = [dist_input, direc_input, elevation_input, day_input, hour_input, dew_point_input]
    knn_X = np.array(knn_features).reshape(1, -1)

    # Convert knn_X to float64
    knn_X = knn_X.astype(np.float64)

    # Load the data
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

    X = dataset[features]
    y = dataset['AIR_TEMPERATURE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN model prediction
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train, y_train)
    knn_prediction = knn_model.predict(knn_X)

    st.write(f"Predicted Air Temperature (KNN): {knn_prediction[0]:.2f} °C")
    st.write("It is important to remember that there is no single interpolation method that can be applied to all situations. Some are more exact and useful than others but take longer to calculate. They all have advantages and disadvantages. In practice, selection of a particular interpolation method should depend upon the sample data, the type of surfaces to be generated and tolerance of estimation errors. For our case KNN worked best with our data and the caching mechanism works by creating a distance matrix that stores the pairwise distances between all points in the dataset. This matrix is computed once during the initialization phase of the algorithm and is then reused for subsequent queries.")
    


def main():
    st.sidebar.title("Select Page")
    selected_page = st.sidebar.selectbox("Choose a page", ["Overview", "Data Analysis", "Now You Try"])

    if selected_page == "Overview":
        page_overview()
    elif selected_page == "Data Analysis":
        page_data_analysis()
    elif selected_page == "Now You Try":
        page_now_you_try()


if __name__ == "__main__":
    main()
