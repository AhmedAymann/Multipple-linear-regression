#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data 

dataset = pd.read_csv('C:/Users/My Pc/Desktop/machine learning tests/linear regression/multiple linear regression/50_Startups.csv')
dp = dataset.iloc[:, :-1].values 
indp = dataset.iloc[:, 4].values


# splitting data into trainingset and testset

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( dp, indp, test_size= 0.2, random_state= 0)

# encoding the independant variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
dp[:, 3] = labelencoder_x.fit_transform(dp[:, 3])
ct = ColumnTransformer(
    [('onehotencoder' , OneHotEncoder(categories = 'auto'), [3])],
    remainder = 'passthrough'
    )
dp = ct.fit_transform(dp)

# avoiding dummy variable trap 
dp = dp[:, 1:] 




# UNKNOWN PROBLEM SOLUTION
x_train, x_test, y_train, y_test = train_test_split( dp, indp, test_size= 0.2, random_state= 0)




# fitting simple linear regretion to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Predecting the test set results 
y_pred = regressor.predict(x_test)


# building the optimal model using backward elimination

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
dp = np.append(arr = np.ones((50, 1)).astype(int), values = dp, axis = 1 ) # adding the coulumn of constant value (1)
x_opt = dp[:, [0, 1, 2, 3, 4, 5]]
x_opt = np.array(x_opt, dtype=float)
regressor_ols = sm.OLS(indp, x_opt).fit()
regressor_ols.summary()
x_opt = dp[:, [0, 1, 3, 4, 5]]
x_opt = np.array(x_opt, dtype=float)
regressor_ols = sm.OLS(indp, x_opt).fit()
regressor_ols.summary()
x_opt = dp[:, [0, 1, 3, 5]]
x_opt = np.array(x_opt, dtype=float)
regressor_ols = sm.OLS(indp, x_opt).fit()
regressor_ols.summary()
x_opt = dp[:, [0, 3, 5]]
x_opt = np.array(x_opt, dtype=float)
regressor_ols = sm.OLS(indp, x_opt).fit()
regressor_ols.summary()
x_opt = dp[:, [0, 3,]]
x_opt = np.array(x_opt, dtype=float)
regressor_ols = sm.OLS(indp, x_opt).fit()
regressor_ols.summary()