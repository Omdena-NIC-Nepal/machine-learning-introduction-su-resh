import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def my_outlier_winsorization(df):
    """
    This function use Winsorization to handle outlier. 
    It is the technique to handle outliers using percentile method. 
    In this method we define a confidence interval of let's say 90% 
    and then replace all the outliers below the 5th percentile with the value at 5th percentile 
    and all the values above 95th percentile with the value at the 95th percentile. 
    """
    # select numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        df[col] = winsorize(df[col], limits=[0.05, 0.05],inclusive=(True, True), inplace=True)
    return df

def feature_selection_rfr(X_train, y_train, n_features=5):
    """
    This method perform feature selection using RandomForestRegressor.
    and returns top n_features based on feature_importances_.
    """
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train, y_train)
#     y_pred = rfr.predict(X_test)
    rfr_importances = pd.Series(rfr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    rfr_importances.plot(kind='bar', title="Feature Ranking:")
    plt.show()     
    print(f'Top {n_features} important features are:')
    print(rfr_importances.head(5))
    return rfr_importances.head(5)

## ------- data preprocessing
## Load the dataset
directory = '../data'
file_name = 'boston_housing.csv'
df = pd.read_csv(os.path.join(directory,file_name))

## Handling outliers
df = my_outlier_winsorization(df)

## Split the data into training and testing sets
X = df.drop(columns='medv')
y = df['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2, random_state=42)

## Feature Scaling (Normalize/standardize numerical features)
scaler = StandardScaler()
# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)
# Transform train and test set
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Convert the result (NumPy array) back to a DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
###------------------------------------------------

##---- Top-5 Features selection using Random Forest Regressor
imp_features = feature_selection_rfr(X_train_scaled_df, y_train, n_features=5)

# Select only top-5 important features.
X_train_selected = X_train_scaled_df[imp_features.index]
X_test_selected = X_test_scaled_df[imp_features.index]
## -------------------------------------------

##--- Train the Linear Regression model
## initilize the linear regression model
lr_model = LinearRegression()

## fit the model
lr_model.fit(X_train_selected, y_train)

## Parameters of Linear regression
print("Coefficient of liner regression are:")
print(lr_model.coef_)
##----------------------------------------------



