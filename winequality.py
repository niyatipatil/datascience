# Wine Quality dataset is used for Classification Problem
# By Niyati P.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

df=pd.read_csv("D:/dataset/winequality.csv")

#-----Checking for missing values-----
print('Missing Values:')
print(df.isnull().sum())

# Fill missing values with mean
df['fixed acidity'].fillna(df['fixed acidity'].mean(), inplace=True)
df['volatile acidity'].fillna(df['volatile acidity'].mean(), inplace=True)
df['citric acid'].fillna(df['citric acid'].mean(), inplace=True)
df['residual sugar'].fillna(df['residual sugar'].mean(), inplace=True)
df['chlorides'].fillna(df['chlorides'].mean(), inplace=True)
df['pH'].fillna(df['pH'].mean(), inplace=True)
df['sulphates'].fillna(df['sulphates'].mean(), inplace=True)

# Verify if missing values have been filled
print('\nAfter filling the Missing Values:')
print(df.isnull().sum())

# Separate features and target variable
X = df.drop('quality', axis=1)
Y = df['quality']

# Convert target variable into categorical classes
# For example, if quality > 6, class = 'Good', else class = 'Bad'
Y = ['Good' if q > 6 else 'Bad' for q in Y]

print("This are the attributes of the dataset:\n",X)
print("This is the target values of the dataset:\n",Y)


from sklearn.preprocessing import OneHotEncoder
# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X)

#-----Training and Testing the model-----
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, random_state=1, test_size=0.2)

#-----RandomForestClassifier-----
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, Y_train)    #Train the model using the training data

y_pred = rf.predict(X_test)  #Predict on the testing data

accuracy = accuracy_score(Y_test, y_pred)   #Calculate the accuracy of the model
print("Accuracy using Random Forest Classifier:", accuracy)

#-----LogisticRegression-----

lr = LogisticRegression(random_state=1)
lr.fit(X_train,Y_train)

y_pred=lr.predict((X_test))
print('Accuracy using Logistics Regression:',accuracy_score(Y_test,y_pred))


#-----DecisionTreeClassifier-----

dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, Y_train)
y_pred = dt.predict(X_test)
print('Accuracy using Decision Tree Classifier:',accuracy_score(Y_test,y_pred))


#-----GradientBoostingClassifier-----

gbm = GradientBoostingClassifier(n_estimators=10)
gbm.fit(X_train, Y_train)
y_pred = gbm.predict(X_test)
print('Accuracy using Gradient Boosting Classifier:',accuracy_score(Y_test,y_pred))




# You can also explore the process of feature selection to identify the most relevant features for your models
# feature selection code here
'''
# Separate features and target variable
X = df.drop('quality', axis=1)
Y = df['quality']

from sklearn.preprocessing import OneHotEncoder
# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X)
#print(X_encoded.columns.values)


from sklearn.feature_selection import SelectKBest #SelectKBest is a package with several functions
from sklearn.feature_selection import chi2 #chi2 for chi square test

bestfeatures = SelectKBest(score_func=chi2, k='all') #k=all means for all values
fit = bestfeatures.fit(X_encoded,Y) #.fit means training the model
dfscores = pd.DataFrame(fit.scores_)  #storing the scores in df (imp work done here other is for understanding)
dfcolumns = pd.DataFrame(X_encoded.columns)  #df values stored
featureScores = pd.concat([dfcolumns, dfscores], axis=1) #scores and cols concatenate
featureScores.columns = ['Features', 'Score'] #label the 2 cols

print(featureScores)

X_enc = X_encoded.drop('fixed acidity', axis=1)
X_enc = X_enc.drop('volatile acidity', axis=1)
X_enc = X_enc.drop('citric acid', axis=1)
X_enc = X_enc.drop('chlorides', axis=1)
X_enc = X_enc.drop('density', axis=1)
X_enc = X_enc.drop('pH', axis=1)
X_enc = X_enc.drop('sulphates', axis=1)
print(X_enc.columns.values)

# Convert target variable into categorical classes
# For example, if quality > 6, class = 'Good', else class = 'Bad'
Y = ['Good' if q > 6 else 'Bad' for q in Y]

print("This are the attributes of the dataset:\n",X)
print("This is the target values of the dataset:\n",Y)

#-----Training and Testing the model-----
X_train, X_test, Y_train, Y_test = train_test_split(X_enc, Y, random_state=1, test_size=0.2)

#-----RandomForestClassifier-----
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, Y_train)    #Train the model using the training data
y_pred = rf.predict(X_test)  #Predict on the testing data
accuracy = accuracy_score(Y_test, y_pred)   #Calculate the accuracy of the model
print("Accuracy using Random Forest Classifier:", accuracy)

#-----LogisticRegression-----
lr = LogisticRegression(random_state=1)
lr.fit(X_train,Y_train)
y_pred=lr.predict((X_test))
print('Accuracy using Logistics Regression:',accuracy_score(Y_test,y_pred))


#-----DecisionTreeClassifier-----
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, Y_train)
y_pred = dt.predict(X_test)
print('Accuracy using Decision Tree Classifier:',accuracy_score(Y_test,y_pred))


#-----GradientBoostingClassifier-----
gbm = GradientBoostingClassifier(n_estimators=10)
gbm.fit(X_train, Y_train)
y_pred = gbm.predict(X_test)
print('Accuracy using Gradient Boosting Classifier:',accuracy_score(Y_test,y_pred))
'''