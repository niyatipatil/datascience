# The goal of the Census Income dataset is to develop classification models that can predict the income level of individuals
# based on their demographic and socio-economic attributes.
# By Niyati P.

# The dataset may contain missing values, denoted as "?", in certain feature columns.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('D:/dataset/census_income.csv')   #Replace 'census_income.csv' with the actual filename

#df['age'] = pd.to_numeric(df['age'], errors='coerce')

#-----Checking for missing values-----
print('Missing Values:')
print(df.isnull().sum())

# Convert the target variable to numerical values
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})                #less than or equal to 50K USD is 0 & greater than 50K USD is 1
print('\nConverting income to distinct value:')
print(df['income'])

# Perform one-hot encoding on categorical columns
categorical_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'education']
data_encoded = pd.get_dummies(df, columns=categorical_cols)
print(data_encoded)
#print(data_encoded.dtypes)

# Define the feature columns and target variable
feature_cols = data_encoded.columns.drop('income')
#feature_cols = data_encoded.columns.drop('income','education')
target_col = 'income'

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data_encoded[feature_cols], data_encoded[target_col], random_state=1, test_size=0.2)

# Create and train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy using logistic regression:', accuracy)

#Trying other classification Algorithms
#-----DecisionTreeClassifier-----
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, Y_train)
y_pred = dt.predict(X_test)
print('Accuracy using Decision Tree Classifier:',accuracy_score(Y_test,y_pred))

#-----GradientBoostingClassifier-----
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=10)
gbm.fit(X_train, Y_train)
y_pred = gbm.predict(X_test)
print('Accuracy using Gradient Boosting Classifier:',accuracy_score(Y_test,y_pred))

#NOTE: The Accuracy of the Model is removed based on all the features
