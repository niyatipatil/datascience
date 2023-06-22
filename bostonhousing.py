# This code performs various data science tasks on the boston housing dataset.
# It includes loading the dataset from a CSV file, preparing the data, and training and testing a linear regression model.
# By Niyati P.

#---------1. CONSIDERING ALL THE FEATURES---------
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("D:/dataset/bostonhousing.csv")

#-----Checking for missing values-----
print('Missing Values:')
print(df.isnull().sum())
print(df.notnull().sum())

#-----Selecting features and Target variables-----
X = df.drop('MEDV', axis=1)         #Features
Y = df['MEDV']                      #Target variable
print("This are the attributes of the dataset:\n",X)
print("This are the target values of the dataset:\n",Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.2)


print('\nLinear Regression')
lreg = LinearRegression()
lreg.fit(X_train,Y_train)

y_pred = lreg.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:",mse)
r2 = r2_score(Y_test, y_pred)
print("R-squared Score:", r2)

#-----using other algorithms-----

print('\nGradient Boosting Regressor')
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train,Y_train)

y_pred = gbr.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:",mse)
r2 = r2_score(Y_test, y_pred)
print("R-squared Score:", r2)

print('\nDecision Tree Regressor')
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(X_train,Y_train)

y_pred = dtr.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:",mse)
r2 = r2_score(Y_test, y_pred)
print("R-squared Score:", r2)
