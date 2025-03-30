import pandas as pd
import os

import numpy as np

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def detect_outliers(df):
    """
    This method use IQR method to detect outliers.
    """
    outliers_dict = {}
    # select numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    outliers = pd.DataFrame(columns = ['Feature', 'Number of Outliers'])
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - (1.5*iqr)
        fence_high = q3 + (1.5*iqr)
        df_outliers_col = df.loc[(df[col] < fence_low) | (df[col] > fence_high), [col]]
        outliers_dict[col] = len(df_outliers_col)
        outliers_df = pd.DataFrame(outliers_dict.items(), columns=['Feature', 'Number of outliers'])
    return outliers_df

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

def my_identify_categorical_features(df, threshold):
    """
    This method uses thereshold for identifucation of categorical feature
    If count of uniques value in feature is less than threshold, then that feature is considered as categorical
    """
    # If unique values < threshold, assume it is categorical
    categorical_columns = [col for col in df.columns if df[col].nunique() < threshold]
    print("{} Categorical Feature(s):".format(len(categorical_columns)), categorical_columns)
    return categorical_columns

## Load the dataset
directory = '../data'
file_name = 'boston_housing.csv'
df = pd.read_csv(os.path.join(directory,file_name))

##--- Handling missing values
print('The count of null values in each column of the dataset are as follows:')
df.isnull().sum()
### Since there is no missing values in the dataset, there is no need to handle missing value
##-------------------------------------------------

##--- Handling outliers
outlier_df = detect_outliers(df)
print("Count of outliers from IQR rule before handling outliers:")
print(outlier_df)

df = my_outlier_winsorization(df)

outlier_df=detect_outliers(df)
print("Count of outliers from IQR rule after handling outliers:")
print(outlier_df)
##-------------------------------------

##--- Encode categorical variables
categoical_columns = my_identify_categorical_features(df, 5)
print('Values of categorical variable:')
print(df[categoical_columns].value_counts())
## Here with threshold = 5, 'chas' feature of given dataset is found to be categorical. 
# This feature has already numerical values, which can be considered as label encoding. 
# So, no seperate encoding for this categorical variable is considered.
##---------------------------------------------

##--- Split the data into training and testing sets
## Split dataset into Features (X) and target (y)
X = df.drop(columns='medv')
y = df['medv']
## split data into tran-test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2, random_state=42)
##--------------------------------------------------

##--- Feature Scaling (Normalize/standardize numerical features)
scaler = StandardScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# Transform train and test set
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
###----------------------------------------------------

