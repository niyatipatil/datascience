# Linear Regression on Black Friday Sales to predict values
# By Niyati P.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('D:/dataset/blackfridaysales/train.csv')
test_df = pd.read_csv('D:/dataset/blackfridaysales/test.csv')

#-----Checking for missing values-----
print('Missing Values in training dataset:')
print(train_df.isnull().sum())
print('Missing Values in testing dataset:')
print(test_df.isnull().sum())
#print(train_df.notnull().sum())

train_df['Product_Category_2'].fillna(train_df['Product_Category_2'].mean(), inplace=True)
train_df['Product_Category_3'].fillna(train_df['Product_Category_3'].mean(), inplace=True)

test_df['Product_Category_2'].fillna(test_df['Product_Category_2'].mean(), inplace=True)
test_df['Product_Category_3'].fillna(test_df['Product_Category_3'].mean(), inplace=True)
print('\nRechecking the Missing Values in both the dataset:')
print(train_df.isnull().sum())
print(test_df.isnull().sum())

#-----Checking the columns in dataset
#print("Train Dataset Columns:", train_df.columns)
#print("Test Dataset Columns:", test_df.columns)

combined_df = pd.concat([train_df, test_df])    #Combine train and test datasets for label encoding
print(combined_df)

#Apply label encoding to categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
for col in categorical_cols:
    combined_df[col] = le.fit_transform(combined_df[col])

train_df = combined_df[:len(train_df)]      #Split the combined dataset back into train and test datasets
test_df = combined_df[len(train_df):]

#training the both the datasets
X_train = train_df.drop('Purchase', axis=1)  # Features for training
X_train = X_train.drop('Product_ID', axis=1)

y_train = train_df['Purchase']  # Target variable for training
#dataset does not include the "Purchase" column, it means you need to make predictions for the target variable using the trained regression model.

#testing the dataset using linear regression model
lreg = LinearRegression()
lreg.fit(X_train,y_train)

X_test = test_df.drop('Purchase', axis=1)
X_test = X_test.drop('Product_ID', axis=1)
y_pred = lreg.predict(X_test)

# Print the predicted purchase amounts
print("Predicted Purchase Amounts:")
print(y_pred)




#-----------------------------------------------------------------------------------------------
'''
#Linear Regression on Black Friday Sales to predict values removing Product_Category_2 and 3

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('D:/driveB/internship_23/dataset/blackfridaysales/train.csv')
test_df = pd.read_csv('D:/driveB/internship_23/dataset/blackfridaysales/test.csv')

#-----Checking for missing values-----
print('Missing Values:')
print(train_df.isnull().sum())
#print(train_df.notnull().sum())

#-----Checking the columns in dataset
print("Train Dataset Columns:", train_df.columns)
print("Test Dataset Columns:", test_df.columns)

combined_df = pd.concat([train_df, test_df])    #Combine train and test datasets for label encoding
print(combined_df)

#Apply label encoding to categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
for col in categorical_cols:
    combined_df[col] = le.fit_transform(combined_df[col])

train_df = combined_df[:len(train_df)]      #Split the combined dataset back into train and test datasets
test_df = combined_df[len(train_df):]

#training the both the datasets
X_train = train_df.drop('Purchase', axis=1)  # Features for training
X_train = X_train.drop('Product_Category_2', axis=1)
X_train = X_train.drop('Product_Category_3', axis=1)
X_train = X_train.drop('Product_ID', axis=1)

y_train = train_df['Purchase']  # Target variable for training
#dataset does not include the "Purchase" column, it means you need to make predictions for the target variable using the trained regression model.

#testing the dataset using linear regression model
lreg = LinearRegression()
lreg.fit(X_train,y_train)

X_test = test_df.drop('Purchase', axis=1)
X_test = X_test.drop('Product_Category_2', axis=1)
X_test = X_test.drop('Product_Category_3', axis=1)
X_test = X_test.drop('Product_ID', axis=1)
y_pred = lreg.predict(X_test)

# Print the predicted purchase amounts
print("Predicted Purchase Amounts:")
print(y_pred)
'''
