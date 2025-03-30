import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

###----- load and pre-processig dataset
# load data
directory = '../data'
file_name = 'boston_housing.csv'
df = pd.read_csv(os.path.join(directory,file_name))

## Outliers handling with Winsorization method.
## This method uses winsorization to handle outliers 
## where the lowest 5% and highest 5% of values are replaced by value at corresponding percentiles (5th and 95th).
def my_outlier_winsorization(df):
    # select numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        df[col] = winsorize(df[col], limits=[0.05, 0.05],inclusive=(True, True), inplace=True)
    return df
df = my_outlier_winsorization(df)

## Split dataset into Features (X) and target (y)
X = df.drop(columns='medv')
y = df['medv']
## split data into tran-test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2, random_state=42)

## perform Normalize/standardize using StandardScaler()
scaler = StandardScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# Transform train and test set
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the result (NumPy array) back to a DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

def feature_selection_rfr(X_train, y_train, n_features=5):
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train, y_train)
#     y_pred = rfr.predict(X_test)
    rfr_importances = pd.Series(rfr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    rfr_importances.plot(kind='bar', title="Feature Ranking:")
    plt.show()     
    print(f'Top {n_features} important features are:')
    print(rfr_importances.head(n_features))
    return rfr_importances.head(n_features)

imp_features=feature_selection_rfr(X_train_scaled_df, y_train)

# Select only top-5 important features.
X_train_selected = X_train_scaled_df[imp_features.index]
X_test_selected = X_test_scaled_df[imp_features.index]
###--------------------------------------------

##--- Train the Linear Regression model
## initilize the linear regression model
lr_model = LinearRegression()

## fit the model
lr_model.fit(X_train_selected, y_train)

## Parameters of Linear regression
print(lr_model.coef_)
#------------------------------------

##--- Model evaluation
## Make prediction
y_pred=lr_model.predict(X_test_selected)

# Model evaluation using various using metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
mse = metrics.mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
print("RMSE: ", np.sqrt(mse))
r2_score = metrics.r2_score(y_test, y_pred)
print("R^2: ", r2_score)
print("Adusted R^2: ", 1-((1- r2_score)*(len(df)-1))/(len(df)-X_train_selected.shape[1])-1)

##--- Plot residuals to check the assumptions of linear regression
# visualizing the difference between the actual and predicted price in scatter plot
fig = plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=12)
plt.show()

errors = y_test - y_pred
fig = plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=20, kde=True)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Histogram of Errors", fontsize=12)
plt.show()

####----- Compare model performance with different feature sets or preprocessing steps.
## We perform linear regressio with following variations for performance comparision
### Use StandardScaler() and MinMaxScaler() for Normalize/standardize.
### Use top 3, 5 and 7 important features

scaler_dict = {'standard_scaler': StandardScaler(), 
           'minmax_scaler': MinMaxScaler()}
n_features = [3, 5, 7]

### perform Linear regression model with different scaling and feature selection for performance comparision
performance_dict_lst = []
for scaler_key, scaler_value in scaler_dict.items():
    for n in n_features:
        print(f'{scaler_key}_{n}')
        
        ## perform scaling
        scaler = scaler_value # select scaler
        # fit on train set
        scaler.fit(X_train)
        # transform train and test set
        X_train_scaled = scaler.transform(X_train) 
        X_test_scaled = scaler.transform(X_test)
        # Convert the result (NumPy array) back to a DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        ## feature selection
        imp_features=feature_selection_rfr(X_train_scaled_df, y_train, n_features= n)
        # Select only top-5 important features.
        X_train_selected = X_train_scaled_df[imp_features.index]
        X_test_selected = X_test_scaled_df[imp_features.index]
        
        # Train the Linear Regression model
        ## initilize the linear regression model
        lr_model = LinearRegression()
        ## fit the model
        lr_model.fit(X_train_selected, y_train)

        ## Make prediction
        y_pred=lr_model.predict(X_test_selected)
        
        ## calculate R^2 score
        r2_score = metrics.r2_score(y_test, y_pred)
        
        ## Record the R^2 score
        performance_dict = {
            'Standardization Scaler': scaler_key,
            'Number of Features': n,
            'R^2 Score': r2_score
        }
        performance_dict_lst.append(performance_dict)
performance_df = pd.DataFrame(performance_dict_lst)

print(performance_df)
