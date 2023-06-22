# Linear Regression on Walmart Sales to predict values
# By Niyati P.

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import numpy as np

train = pd.read_csv("D:/dataset/walmart//train.csv")       #Columns are Store, Dept, Date, Weekly_Sales, IsHoliday
                                                                                #used for training a predictive model
stores = pd.read_csv("D:/dataset/walmart/stores.csv")      #Columns are Store, Dept, Date, IsHoliday
features = pd.read_csv("D:/dataset/walmart/features.csv")  #Columns are Store, Date, Temperature, Fuel_Price, MarkDown1,2,3,4,5, CPI, Unemployment, IsHoliday
test1 = pd.read_csv("D:/dataset/walmart/test.csv")         #Columns are Store, Date, Temperature, Fuel_Price, MarkDown1,2,3,4,5, CPI, Unemployment, IsHoliday
                                                                                #The purpose is to evaluate the model's performance on unseen data; It includes similar information to the training data as well as additional features

# Merge all csv files in one df:
df1 = train.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner') #function is used to combine the train and features
                                                                                #The on parameter specifies the columns to merge on. The how parameter is set to 'inner',which means only the rows that have matching values in all 3 cols in both DataFrames will be included.
df = df1.merge(stores, on = ['Store'], how = 'inner')                           #DataFrame df1 is further merged with the stores DataFrame, this time based on the common column 'Store'
print('Our merged train ds. DataFrame is:')
print(df.head())
print('Missing values in Train ds.:')
print(df.isnull().sum())

# We will drop the Markdown columns as they have null values over 30%:

df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1, inplace = True)    #drop() function is used to remove specific columns(MarkDown1,2,3,4,5) from the DataFrame df.
                                                                                                        #axis = 1: indicate that the operation should be performed along the columns.
                                                                                                        #inplace = True : the changes are made directly to the df DataFrame without creating a new DataFrame.
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(keys = "Date", inplace = True)
print(df.isnull().sum())  #rechecking for null values

# Removing Outliers:
columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']

Q3 = df[columns].quantile(.75)
Q1 = df[columns].quantile(.25)
IQR = Q3 - Q1
UL = Q3 + 1.5*IQR
LL = Q1 - 1.5*IQR

for column in columns:
    df[column] = np.where(df[column] > UL[column], UL[column], np.where(df[column] < LL[column], LL[column], df[column]))

# Data Preparation for Testing dataset; performing soimilar functions as performed on Training dataset:
df_test = test1.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')
test = df_test.merge(stores, on = ['Store'], how = 'inner')
print('\nOur merged test ds. DataFrame is:')
print(test.head())

test.drop(axis = 1, columns = ["MarkDown1", "MarkDown2","MarkDown3","MarkDown4", "MarkDown5"], inplace = True)
print('Missing values in Test ds.:')
print(test.isnull().sum())

# Filling null values with mean
test['CPI'] = test['CPI'].fillna(test['CPI'].mean())
test['Unemployment'] = test['Unemployment'].fillna(test['Unemployment'].mean())
print(test.isnull().sum()) #rechecking for null values


# Removing outliers
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(test['Unemployment'])
plt.show()

columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']

Q3 = test[columns].quantile(.75)
Q1 = test[columns].quantile(.25)
IQR = Q3 - Q1
UL = Q3 + 1.5*IQR
LL = Q1 - 1.5*IQR

for column in columns:
    test[column] = np.where(test[column] > UL[column], UL[column], np.where(test[column] < LL[column], LL[column], test[column]))

#print(test)
sns.boxplot(test['Unemployment'])           #rechecking if outliers are removed or not?
plt.show()


test['Date'] = pd.to_datetime(test['Date']) #This conversion allows for performing various date-related operations on the column
test.set_index(keys = 'Date', inplace = True)
print(test.head())

# categorical variables into numeric labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['IsHoliday'] = le.fit_transform(df['IsHoliday'])
df['Type'] = le.fit_transform(df['Type'])
test['IsHoliday'] = le.fit_transform(test['IsHoliday'])
test['Type'] = le.fit_transform(test['Type'])

print(df[['IsHoliday', 'Type']])
print(test[['IsHoliday', 'Type']])

df['CPI'] = df['CPI'].round(2)              #'CPI' column should be rounded to two decimal places.
print('CPI to two decimals:\n',df['CPI'])


# splitting the dataset into features (inputs) and target (output).

X = df.drop(['Weekly_Sales'], axis = 1)
Y = df['Weekly_Sales']
print("These are the features of the dataset:\n",X)
print("These are the target values of the dataset:\n",Y)


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state = 1, test_size = 0.2 )

print("X Train Shape :",X_train.shape)  #tuple that represents the dimensions of the array or matrix;[number of samples (rows) and the number of features (columns)] in the training data.
print("X Val Shape   :",X_val.shape)
print("Y Train Shape :",Y_train.shape)
print("Y Val Shape   :",Y_val.shape)


# training a linear regression model, making predictions on the validation set, and evaluating the model's performance
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_model = lr.fit(X_train, Y_train)
y_pred = lr_model.predict(X_val)

lr.score(X_val, Y_val)  #coefficient of determination [R-squared (R^2) score] to evaluate the performance of the trained linear reg. model on the validation data.

mse = mean_squared_error(Y_val, y_pred)
r2 = r2_score(Y_val, y_pred)
print('MSE of LR = ', mse)
print('R2 Score of LR= ', r2)

print("\nPredicted Weekly Sales:")
print(y_pred)

'''
# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt_model = dt.fit(X_train, Y_train)
y_pred_dt = dt_model.predict(X_val)

mse_dt = mean_squared_error(Y_val, y_pred_dt)
r2_dt = r2_score(Y_val ,y_pred_dt)
print('MSE of DT = ', mse_dt)
print('R2 Score of DT = ', r2_dt)
'''

'''
# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_model = rf.fit(X_train, Y_train)
y_pred_rf = rf_model.predict(x_val)

mse_rf = mean_squared_error(y_pred_rf, Y_val)
r2_rf = r2_score(y_pred_rf, Y_val)
print('MSE of RF = ', mse_rf)
print('R2 Score of RF = ', r2_rf)'''