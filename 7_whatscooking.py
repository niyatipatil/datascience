# Applying Classification a Supervised Learning Technique on the WhatsCooking data
# In addition, acquiring knowledge and skills in working with .json files instead of csv
# By Niyati P.

import pandas as pd                                         # data manipulation and analysis
import json                                                 # handle JSON file operations
import numpy as np                                          # for numerical computations
from sklearn.feature_extraction.text import TfidfVectorizer # convert text data into numerical features using TF-IDF
from sklearn.model_selection import train_test_split        # to split the data into training and testing sets
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC                                 # for different classification algorithms
from sklearn.metrics import classification_report, accuracy_score # to evaluate the model's performance

with open("D:/driveB/internship_23/dataset/whatscooking/cooking.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
print('dataset is:\n', df.head(10))
print('Columns in the dataset are:\n', df.columns)
print(df.describe())
print(df['cuisine'].nunique())

#-----Handling Missing Values-----
print('\nMissing values:')
print(df.isnull().sum())
#df.dropna(inplace=True)

# df['ingredients'] = df['ingredients'].apply(','.join)
# df.drop_duplicates(inplace=True)
# print({df.duplicated().sum()})

#-----Handling Empty Documents-----
empty_documents = df[df['ingredients'].apply(lambda x: len(x) == 0)]
if not empty_documents.empty:
    print("Empty documents found!")
    # Handle empty documents here, such as dropping or filling them
else:
    print("No empty documents found.")

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df['ingredients'].apply(' '.join))
print(features)


#-----Model training and evaluation-----

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(features, df['cuisine'], test_size=0.2, random_state=42)

# (Multinomial Naive Bayes)
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("----------------------------------------------------------")
print("Multinomial Naive Bayes Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("------------------------------------------------------")
print("Logistic Regression Accuracy:", accuracy_lr)
print(classification_report(y_test, y_pred_lr))

# You can also try for other classification techniques
# Use the code below
'''
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("-----------------------------------------------------")
print("Random Forest Accuracy:", accuracy_rf)
print(classification_report(y_test, y_pred_rf))

# Support Vector Machines (SVM)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("-----------------------------------------------------")
print("SVM Accuracy:", accuracy_svm)
print(classification_report(y_test, y_pred_svm))
'''
