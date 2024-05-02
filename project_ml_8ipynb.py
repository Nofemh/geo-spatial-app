

#step 1: Import Python Libraries
import numpy as np
import pandas as pd
import pyproj
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import geopandas as gpd
from shapely.geometry import Point
from scipy.optimize import minimize
import contextily as ctx
import plotly.express as px
import lightgbm as lgb

#Step 2: Load the Dataset
dataset = pd.read_csv('modified_dataset.csv')
dataset.head()

print("Before feature Engineering:")
dataset.shape

#Step 4: Feature Engineering
# Convert 'OBSERVATION_DATE' to datetime
dataset['OBSERVATION_DATE'] = pd.to_datetime(dataset['OBSERVATION_DATE'])

# Extract day of year and hour
dataset['DAY_OF_YEAR'] = dataset['OBSERVATION_DATE'].dt.dayofyear
dataset['HOUR'] = dataset['OBSERVATION_DATE'].dt.hour

# Calculate distances and bearings from the first coordinate
geodesic = pyproj.Geod(ellps='WGS84')

def get_dist_bearing(p1, p2):
    ''''
    p1: lat,lon
    p2: lat,lon
    in degrees
    '''
    lat1, lon1 = p1
    lat2, lon2 = p2
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return distance, fwd_azimuth

# Extract coordinates from the dataset and convert to a set of tuples
coords = list(zip(dataset['LATITUDE'], dataset['LONGITUDE']))

def geometric_median(points : list):
    points = np.array(points)  # Convert to numpy array for array operations
    # Initial guess for the geometric median (centroid)
    initial_guess = np.mean(points, axis=0)

    # Define the objective function (sum of distances to minimize)
    def objective_function(center):
        return np.sum(np.linalg.norm(points - center, axis=1))

    # Use numerical optimization (minimization) to find the geometric median
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')

    return result.x

center = geometric_median(coords)

# Calculate distance and bearing for each coordinate relative to the center
dataset['dist_bearing'] = dataset.apply(lambda row: get_dist_bearing(center, (row['LATITUDE'], row['LONGITUDE'])), axis=1)

# Extract distance and bearing into separate columns
dataset['DIST'], dataset['DIREC'] = zip(*dataset['dist_bearing'])

#Step 5: Create Features
# Define  features for analysis
features = [
    'DIST',  # Geographic coordinate: Latitude
    'DIREC',  # Geographic coordinate: Longitude
    'ELEVATION',  # Elevation above sea level
    'DAY_OF_YEAR',  # Day of the year (from OBSERVATION_DATE)
    'HOUR',  # Hour of the day (from OBSERVATION_DATE)
    'AIR_TEMPERATURE_DEW_POINT'  # Dew point temperature
]

print("After feature Engineering:")
dataset.shape

#Step 6: Data Cleaning
# Remove duplicate rows
dataset = dataset.drop_duplicates()

#columns = data.columns
#print("Columns in the dataset:")
#for col in columns:
#     print(col)



#Univariate Analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Create histograms for each feature
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=dataset, x=feature)
    plt.title(f"Distribution of {feature}")
    plt.show()

#Bivariate Analysis
# Create scatter plots for pairs of features
for i in range(len(features)):
    for j in range(i+1, len(features)):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=dataset, x=features[i], y=features[j])
        plt.title(f"Relationship between {features[i]} and {features[j]}")
        plt.show()

#Multivariate Analysis
import seaborn as sns

# Correlation matrix
corr_matrix = dataset[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix')
plt.show()

"""-A value of 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.

-The strength of the linear relationship can be measured by the correlation coefficient (r), which ranges from -1 to 1.

-An r value close to 1 or -1 indicates a strong linear relationship, while a value close to 0 indicates a weak or no linear relationship.
"""

#Impute Missing Values
from sklearn.impute import SimpleImputer

# Create an imputer object and fit it to the dataset
imputer = SimpleImputer(strategy='mean')
dataset_imputed = imputer.fit_transform(dataset[features])

#dataset[features].isnull().sum()

# Step 8: Statistics Summary
print("Dataset Statistics:")
dataset.describe()



"""# Developing Machine Learning Models

FIRST ALGORITHM : Perform KNN
"""

# Extract relevant features and target variable
X = dataset[['LONGITUDE', 'LATITUDE', 'ELEVATION', 'DAY_OF_YEAR', 'HOUR']].values
y = dataset['AIR_TEMPERATURE'].values

from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit KNN model for interpolation
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model using  (MSE)
from sklearn.metrics import mean_squared_error
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"Mean Squared Error (KNN): {mse_knn}")

# Evaluate the model using MAE
from sklearn.metrics import mean_absolute_error
mae_knn = mean_absolute_error(y_test, y_pred_knn)
print(f"Mean Absolute Error (KNN): {mae_knn}")


# Evaluate the model using RMSE
rmse_knn = np.sqrt(mse_knn)
print(f"Root Mean Squared Error (KNN): {rmse_knn}")


# Evaluate the model using R-squared (R2)
from sklearn.metrics import r2_score
r2_knn = r2_score(y_test, y_pred_knn)
print(f"R-squared score (KNN): {r2_knn}")

import matplotlib.pyplot as plt
# Scatter plot for actual vs predicted values
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred_knn, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual AIR_TEMPERATURE')
plt.ylabel('Predicted AIR_TEMPERATURE')
plt.title('Scatter plot of Actual vs Predicted AIR_TEMPERATURE (KNN)')
plt.grid()
plt.show()

"""This scatter plot represnt  an idea of how well the KNN model is predicting the AIR_TEMPERATURE values. The closer the points are to the diagonal line (the red dashed line), the better the predictions. If the points deviate significantly from the line, it indicates that the model's predictions are inaccurate.

second model: Train a LightGBM model
"""

X = dataset[features]
y = dataset['AIR_TEMPERATURE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_test])

y_pred_lgbm = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 2-Evaluate the model using  MSE
mse = mean_squared_error(y_test, y_pred_lgbm)
print(f"Mean Squared Error (GBM): {mse}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluate the model using MAE
mae = mean_absolute_error(y_test, y_pred_lgbm)
print(f"Mean Absolute Error (GBM): {mae}")

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f"Root Mean Squared Error (GBM): {rmse}")

# Evaluate the model using R-squared (R2)
r2 = r2_score(y_test, y_pred_lgbm)
print(f"R-squared (GBM): {r2}")

# Plot feature importances
lgb.plot_importance(gbm, figsize=(10, 6))
plt.title("Feature Importances")
plt.show()

"""Theered model: Ridge Regression"""

from sklearn.linear_model import Ridge

# Initialize and fit Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter for regularization
ridge_model.fit(X_train, y_train)

# Predict on the test set
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the Ridge Regression model using Mean Squared Error (MSE)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Mean Squared Error (Ridge Regression): {mse_ridge}")
# Import necessary functions
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate the model using MAE
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
print(f"Mean Absolute Error (Ridge Regression): {mae_ridge}")

# Evaluate the model using RMSE
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"Root Mean Squared Error (Ridge Regression): {rmse_ridge}")

# Evaluate the model using R-squared (R2)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"R-squared (Ridge Regression): {r2_ridge}")

"""


Fourth model : gradient boosting regressor"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Create the gradient boosting regressor
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Fit the model to the training data
gbr.fit(X_train, y_train)

# Make predictions on the test data
y_pred_gbr = gbr.predict(X_test)

# Evaluate the model on the test data
# Evaluate the model using MSE
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print("Mean Squared Error (GBR):", mse_gbr)


# Evaluate the model using MAE
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
print("Mean Absolute Error (GBR):", mae_gbr)


# Evaluate the model using RMSE
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
print(f"Root Mean Squared Error (GBR): {rmse_gbr}")

# Evaluate the model using R-squared (R2)
r2_gbr = r2_score(y_test, y_pred_gbr)
print(f"R-squared (GBR): {r2_gbr}")

"""## **GridSearch**"""

from sklearn.model_selection import GridSearchCV

# 1. KNN

# Define parameter grid
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11]
}

# Create GridSearchCV object
grid_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search
grid_knn.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters for KNN:", grid_knn.best_params_)
print("Best score for KNN:", -grid_knn.best_score_)

# Make predictions on the test data using the best estimator found by grid search
y_pred_knn_tuned = grid_knn.best_estimator_.predict(X_test)

# Evaluate the performance of the tuned KNN model
print("\nPerformance metrics for KNN after tuning:")

# Evaluate Mean Squared Error (MSE)
mse_knn_tuned = mean_squared_error(y_test, y_pred_knn_tuned)
print("Mean Squared Error (MSE):", mse_knn_tuned)

# Evaluate Mean Absolute Error (MAE)
mae_knn_tuned = mean_absolute_error(y_test, y_pred_knn_tuned)
print("Mean Absolute Error (MAE):", mae_knn_tuned)

# Evaluate Root Mean Squared Error (RMSE)
rmse_knn_tuned = np.sqrt(mse_knn_tuned)
print("Root Mean Squared Error (RMSE):", rmse_knn_tuned)

# Evaluate R-squared (R2) Score
r2_knn_tuned = r2_score(y_test, y_pred_knn_tuned)
print("R-squared (R2) Score:", r2_knn_tuned)

# 2. LightGBM

# Define parameter grid
param_grid_lgbm = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

# Create untrained LightGBM model
lgbm_model = lgb.LGBMRegressor()

# Create GridSearchCV object with the untrained model
grid_lgbm = GridSearchCV(lgbm_model, param_grid_lgbm, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search
grid_lgbm.fit(X_train, y_train)

# Make predictions on the test data using the best estimator found by grid search
y_pred_lgbm_tuned = grid_lgbm.best_estimator_.predict(X_test)

#Print best parameters and score
print("Best parameters for LightGBM:", grid_lgbm.best_params_)
print("Best score for LightGBM:", -grid_lgbm.best_score_)

# Evaluate the LightGBM model after tuning

print("Performance metrics for LightGBM after tuning:")

# Evaluate Mean Squared Error (MSE)
mse_lgbm_tuned = mean_squared_error(y_test, y_pred_lgbm_tuned)
print("Mean Squared Error (MSE):", mse_lgbm_tuned)

# Evaluate Mean Absolute Error (MAE)
mas_lgbm_tuned = mean_absolute_error(y_test, y_pred_lgbm_tuned)
print("Mean Absolute Error (MAE):", mas_lgbm_tuned)

# Evaluate Mean Squared Error (RMSE)
rmse_lgbm_tuned = np.sqrt(mse_lgbm_tuned)
print("Root Mean Squared Error (RMSE):", rmse_lgbm_tuned)

# Evaluate R-squared (R2) Score
r2_lgbm_tuned = r2_score(y_test, y_pred_lgbm_tuned)
print("R-squared (R2) Score:", r2_lgbm_tuned)

# 3. Ridge Regression

# Define parameter grid
param_grid_ridge = {
    'alpha': [0.1, 1, 10, 100]
}

# Create GridSearchCV object
grid_ridge = GridSearchCV(ridge_model, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search
grid_ridge.fit(X_train, y_train)

#  make predictions on the test data
y_pred_ridge_tuned = grid_ridge.best_estimator_.predict(X_test)

# Print best parameters and score
print("Best parameters for Ridge Regression:", grid_ridge.best_params_)
print("Best score for Ridge Regression:", -grid_ridge.best_score_)

# Evaluate the Ridge Regression model after tuning
print("Performance metrics for Ridge Regression after tuning:")


# Evaluate Mean Squared Error (MSE)
mse_ridge_tuned = mean_squared_error(y_test, y_pred_ridge_tuned)
print("Mean Squared Error (MSE):", mse_ridge_tuned)

# Evaluate Mean Absolute Error (MAE)
mae_ridge_tuned = mean_absolute_error(y_test, y_pred_ridge_tuned)
print("Mean Absolute Error (MAE):", mae_ridge_tuned)

# Evaluate Root Mean Squared Error (RMSE)
rmse_ridge_tuned = np.sqrt(mse_ridge_tuned)
print("Root Mean Squared Error (RMSE):", rmse_ridge_tuned)

# Evaluate R-squared (R2) Score
r2_ridge_tuned = r2_score(y_test, y_pred_ridge_tuned)
print("R-squared (R2) Score:", r2_ridge_tuned)

# 4. Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
# Define parameter grid
param_grid_gbr = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Create untrained Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor()

# Create GridSearchCV object
grid_gbr = GridSearchCV(gbr, param_grid_gbr, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search
grid_gbr.fit(X_train, y_train)

# Predict on the test set with the best model
y_pred_gbr_tuned = grid_gbr.best_estimator_.predict(X_test)

# Print best parameters and score
print("Best parameters for Gradient Boosting Regressor:", grid_gbr.best_params_)
print("Best score for Gradient Boosting Regressor:", -grid_gbr.best_score_)

# Evaluate the Gradient Boosting Regresso model after tuning
print("\nPerformance metrics for Gradient Boosting Regressor after tuning:")

# Evaluate Mean Squared Error (MSE)
mse_gbr_tuned = mean_squared_error(y_test, y_pred_gbr_tuned)
print("Mean Squared Error (MSE):", mse_gbr_tuned)

# Evaluate Mean Absolute Error (MAE)
mas_gbr_tuned = mean_absolute_error(y_test, y_pred_gbr_tuned)
print("Mean Absolute Error (MAE):", mas_gbr_tuned)

# Evaluate Root Mean Squared Error (RMSE)
rmse_gbr_tuned = np.sqrt(mse_gbr_tuned)
print("Root Mean Squared Error (RMSE):", rmse_gbr_tuned)

# Evaluate R-squared (R2) Score
r2_gbr_tuned = r2_score(y_test, y_pred_gbr_tuned)
print("R-squared (R2) Score:", r2_gbr_tuned)

"""## compare the preformance"""

# 1-KNN model
# Before Grid Search
print("1-KNN model:")
print("Before Grid Search:")
print("Mean Squared Error (KNN):", mse_knn)
print("Mean Absolute Error (KNN):", mae_knn)
print("Root Mean Squared Error (KNN):", rmse_knn)
print("R-squared (KNN):", r2_knn)
# After Grid Search
print("\nAfter Grid Search for KNN:")
print("Mean Squared Error (MSE):", mse_knn_tuned)
print("Mean Absolute Error (MAE):", mae_knn_tuned)
print("Root Mean Squared Error (RMSE):", rmse_knn_tuned)
print("R-squared (R2) Score:", r2_knn_tuned)


# 2-LightGBM model
# Before Grid Search
print("\n2-LightGBM model:")
print("Before Grid Search:")
print("Mean Squared Error (LightGBM):", mse)
print("Mean Absolute Error (LightGBM):", mae)
print("Root Mean Squared Error (LightGBM):", rmse)
print("R-squared (LightGBM):", r2)
# After Grid Search for LightGBM
print("\nAfter Grid Search for LightGBM:")
print("Mean Squared Error (LightGBM):", mse_lgbm_tuned)
print("Mean Absolute Error (LightGBM):", mas_lgbm_tuned)
print("Root Mean Squared Error (LightGBM):", rmse_lgbm_tuned)
print("R-squared (LightGBM):", r2_lgbm_tuned)


# 3- Ridge Regression model
# Before Grid Search
print("\n3-Ridge Regression model:")
print("Before Grid Search:")
print("Mean Squared Error (Ridge Regression):", mse_ridge)
print("Mean Absolute Error (Ridge Regression):", mae_ridge)
print("Root Mean Squared Error (Ridge Regression):", rmse_ridge)
print("R-squared (Ridge Regression):", r2_ridge)
# After Grid Search for Ridge Regression
print("\nAfter Grid Search for Ridge Regression:")
print("Mean Squared Error (MSE):", mse_ridge_tuned)
print("Mean Absolute Error (Ridge Regression):", mae_ridge_tuned)
print("Root Mean Squared Error (Ridge Regression):", rmse_ridge_tuned)
print("R-squared (Ridge Regression):", r2_ridge_tuned)


# 4-Gradient Boosting Regressor model
# Before Grid Search
print("\n4-Gradient Boosting Regressor model:")
print("Before Grid Search:")
print("Mean Squared Error (Gradient Boosting Regressor):", mse_gbr)
print("Mean Absolute Error (Gradient Boosting Regressor):", mae_gbr)
print("Root Mean Squared Error (Gradient Boosting Regressor):", rmse_gbr)
print("R-squared (Gradient Boosting Regressor):", r2_gbr)
# After Grid Search for Gradient Boosting Regressor
print("\nAfter Grid Search for Gradient Boosting Regressor:")
print("Mean Squared Error (MSE):", mse_gbr_tuned)
print("Mean Absolute Error (MAE):", mas_gbr_tuned)
print("Root Mean Squared Error (RMSE):", rmse_gbr_tuned)
print("R-squared (R2) Score:", r2_gbr_tuned)

"""1. KNN Model:
Before Grid Search:
MSE: 981.12
MAE: 5.806
RMSE: 31.323
R-squared: -63.822
After Grid Search:
MSE: 2153.23
MAE: 4.195
RMSE: 46.403
R-squared: -141.263
2. LightGBM Model:
Before Grid Search:
MSE: 290.17
MAE: 7.097
RMSE: 17.034
R-squared: -18.172
After Grid Search:
MSE: 796.48
MAE: 8.129
RMSE: 28.222
R-squared: -51.623
3. Ridge Regression Model:
Before Grid Search:
MSE: 2158.99
MAE: 8.066
RMSE: 46.465
R-squared: -141.644
After Grid Search:
MSE: 2158.86
MAE: 8.064
RMSE: 46.464
R-squared: -141.636
4. Gradient Boosting Regressor Model:
Before Grid Search:
MSE: 9541.95
MAE: 10.673
RMSE: 97.683
R-squared: -629.435
After Grid Search:
MSE: 9578.10
MAE: 10.500
RMSE: 97.868
R-squared: -631.824
"""



# extra kriging algo but will only work on vs code :
# Perform Kriging interpolation
# Extract relevant features and target variable
X = dataset[['LONGITUDE', 'LATITUDE', 'ELEVATION', 'DAY_OF_YEAR', 'HOUR']].values
y = dataset['AIR_TEMPERATURE'].values

# Initialize and fit Ordinary Kriging model
ok_model = OrdinaryKriging(
    X[:, 1],            # Longitude (or x-coordinate)
    X[:, 0],            # Latitude (or y-coordinate)
    y,                  # Target variable (air temperature)
    variogram_model= 'linear',  # Variogram model (e.g., 'linear', 'power', 'spherical')
    verbose=False       # Suppress verbose output
)


 # Generate grid points for prediction within specified bounds (Saudi Arabia)
xmin, xmax = 34, 56  # Longitude bounds
ymin, ymax = 15, 33  # Latitude bounds
grid_resolution = 0.2
x_grid = np.arange(xmin, xmax, grid_resolution)
y_grid = np.arange(ymin, ymax, grid_resolution)
xx, yy = np.meshgrid(x_grid, y_grid)

# Interpolate values at grid points
z_interp, sigma_interp = ok_model.execute('grid', x_grid, y_grid)

# Create a DataFrame for the interpolated values
df_interp = pd.DataFrame({
    'Longitude': xx.ravel(),
    'Latitude': yy.ravel(),
    'Interpolated Value': z_interp.flatten()
})

# Create an interactive heatmap using plotly.express
fig = px.density_mapbox(
    df_interp,
    lon='Longitude',
    lat='Latitude',
    z='Interpolated Value',
    radius=10,
    center=dict(lat=np.mean(yy), lon=np.mean(xx)),
    zoom=5,
    mapbox_style="carto-positron",
    title='Kriging Interpolation of Air Temperature in Saudi Arabia'
)
fig.show()