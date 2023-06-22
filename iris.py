# This code performs various data science tasks on the Iris dataset.
# It includes loading the dataset from a CSV file, preparing the data, and training and testing a logistic regression model.
# By Niyati P.

import pandas as pd     #pd is an alias

df = pd.read_csv("D:/driveB/internship_23/Iris.csv")

'''
print("\n------------ Handling Missing Values ------------------\n")
#dealing with missing values if any present
print(df)                       #shows NaN for missing values
print(df.isnull())              #shows true where there is missing value
print('Missing values count:')
print(df.isnull().sum())        #shows count of missing values
#print(df.notnull().sum())
#EG. If the SepalLengthCm column has some missing values, perform the following
df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean(), inplace=True)
print(df)
print(df.isnull().sum())        #rechecking for missing values
'''
print("\n------------ features and target variables ------------------")
#Prepare the data; extract features and target variable
X = df.drop('Id',axis=1)        #Id plays no role in determining the species to drop the colummn
X = X.drop('Species', axis=1)   #Here instead of df.drop used X.drop as it will remove Species along with Id
                                # axis=1 is complete column and axis=0 is one row

Y = df['Species']   #target value

print("This are the attributes of the dataset:\n", X)
print("This are the target values of the dataset:\n", Y)

'''
print("\n------------ Balancing the dataset ------------------")
#balancing dataset with oversampling and undersampling
#suppose classes A, B, and C are represented by 50, 100, and 150 respectively
#oversampling-150,150,150; undersampling-50,50,50 (here randomly removes data, imp data might be lost)
#SMOTE does not create duplicate;one value inbetween dist. of two sample is selected

print("\nTotal count of Species in the dataset:")
se =(df['Species'] == 'Iris-setosa').sum()
print('Iris-setosa:',se)

ver =(df['Species'] == 'Iris-versicolor').sum()
print('Iris-versicolor:',ver)

vir =(df['Species'] == 'Iris-virginica').sum()
print('Iris-virginica:',vir)

#EG. from here we if 2 values IRIS-VIRGINIA are less
#method 1
from collections import Counter
print(Counter(Y)) #50,50,48
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x, y = ros.fit_resample(X,Y)
print(Counter(y)) #after applying RandomOverSampler we get equal counts and a balanced ds
#----------
#method 2
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x, y = ros.fit_resample(X,Y)
print(y)
from collections import Counter
print(Counter(y))
#----------
#method 3 
from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
x, y =sms.fit_resample(X,Y)
print(y)
from collections import Counter
print(Counter(y))
'''

print("\n------------ Feature Selection ------------------\n")

print("*****1.Using SelectKBest*****\n")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2              #chi2 for chi square test

bestfeatures = SelectKBest(score_func=chi2, k='all')    #k=all means for all values
fit = bestfeatures.fit(X,Y)                             #.fit() used for training the model
dfscores = pd.DataFrame(fit.scores_)                    #storing the scores in df (imp work done here other is for understanding)
dfcolumns = pd.DataFrame(X.columns)                     #df values stored
featureScores = pd.concat([dfcolumns, dfscores], axis=1) #scores and columns concatenate
featureScores.columns = ['Specs', 'Score']              #label the 2 columns

print(featureScores)                                    #PetalLength and PetalWidth play an important role as their scores are high


print("\n*****2.Using ExtraTreesClassifier*****\n")

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='barh')           #barh for plotting horizontal bargraph
plt.show()


print("\n-----numerical to categorical using LabelEncoder-----\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

rf = RandomForestClassifier()

df['SepalLengthCm'] = pd.cut(df['SepalLengthCm'], 3, labels=['0','1','2'])
df['SepalWidthCm'] = pd.cut(df['SepalWidthCm'], 3, labels=['0','1','2'])
df['PetalLengthCm'] = pd.cut(df['PetalLengthCm'], 3, labels=['0','1','2'])
df['PetalWidthCm'] = pd.cut(df['PetalWidthCm'], 3, labels=['0','1','2'])
print(df)                                           #Category divided into 3 types and labelled as 0,1,2
                                                    #Centroid for each category is calculated and data instances are distributed...Similar to K-Means.

x = df.drop('Id', axis=1)
x = x.drop('Species', axis=1)
y = df['Species']
#print(y)
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)
print(Y)


#Feature reduction using PCA : dealing with many dimensions
#2 feature values one principal component is found

print('\nPRINCIPAL COMPONENT ANALYSIS')

from sklearn.decomposition import PCA

'''X = df.drop('Id',axis=1)                           
X = X.drop('Species', axis=1)                           
Y = df['Species']'''

pca = PCA(n_components=2)                               #n_components used to decide the number of features required
pca.fit(x)
x = pca.transform(x)
print(x)

print('\n------Model Fitting and Training------')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split    #the data is separated into training and test sets using the traintestsplit function
                                                        #checked whether particular algorithm fits the model using the given features 'x' and target 'y'
                                                        #shuffles the data and then splits it into training and testing datasets in the ratios of 70:30, 80:20, or 90:10.
from sklearn.metrics import accuracy_score              #as supervised learning, the accuracy of the predicted values is evaluated against the actual values.

X_train, X_test, y_train, y_test = train_test_split(x,Y, random_state=0, test_size=0.3) #random state is used to get the same result every time you run the prog. tes_size=0.3 -> 30% test and 70% training

logr = LogisticRegression()
logr.fit(X_train,y_train)

y_pred = logr.predict((X_test))
accuracy = accuracy_score(y_test,y_pred)
print("\nAccuracy of logistic regression model is:", accuracy)
