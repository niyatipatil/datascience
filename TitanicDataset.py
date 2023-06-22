#Classification algorithms on the Titanic dataset
#By Niyati P.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv("D:/dataset/Titanic.csv")

#-----Checking for missing values-----S
print('Missing Values:')
print(df.isnull().sum())
#print(df.notnull().sum())

#-----dealing with missing values-----
df['Age'].fillna(df['Age'].mean(), inplace=True)
#print(df)
#print(df.isnull().sum())

missing_embarked = df["Embarked"].isnull()  #dropped rows of Embarked with null values as station cannot be filled randomly
df = df.drop(df[missing_embarked].index)
print(df.isnull().sum())

#-----Selecting the required features and dropping others-----
X = df.drop('PassengerId',axis=1)
X = X.drop('Name', axis=1)
X = X.drop('SibSp', axis=1)
X = X.drop('Parch', axis=1)
X = X.drop('Ticket', axis=1)
X = X.drop('Fare', axis=1)
X = X.drop('Cabin', axis=1)
X = X.drop('Survived', axis=1)

Y = df['Survived'] #target value
print("This are the attributes of the dataset:\n",X)
print("This are the target values of the dataset:\n",Y)


#-----Categorical values using LabelEncoder-----
le = LabelEncoder()
le.fit(X['Sex'])
X['Sex']=le.transform(X['Sex'])
print(X['Sex']) #Male=1 and Female=0

le = LabelEncoder()
le.fit(X['Embarked'])
X['Embarked']=le.transform(X['Embarked'])
print(X['Embarked'])


#-----Training and Testing the model-----

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.2)

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

#-----MultinomialNB-----

nb = MultinomialNB()
nb.fit(X_train, Y_train)
y_pred = nb.predict(X_test)
print('Accuracy using MultinomialNB:',accuracy_score(Y_test,y_pred))


