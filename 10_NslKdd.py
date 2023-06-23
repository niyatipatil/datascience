# Applying Classification a Supervised Learning Technique on the NSL KDD data
# By Niyati P.

# importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

df = pd.read_csv("D:/driveB/internship_23/kdd_train.csv")

print(df.head(10))

print('Missing Values:')
print(df.isnull().sum())

print('\nShape of the DataFrame:\n(rows,columns)')
print(df.shape)

print(df.describe())

print('\nColumns in the dataset are:')
print(df.columns)

print(df["labels"].value_counts())

# Converting labels to binary classification
df.loc[df.labels != 'normal','labels'] = 'attack'
counts = df['labels'].value_counts()

# converts the categorical values into numerical representations
categorical_cols = ['protocol_type','service', 'flag']
le = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)

# Data imbalancing checking; by creating a pie chart to visualize the class distribution
colors = ['red', 'blue']
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors)
plt.title('CLASS DISTRIBUTION')
plt.show()

# Checking and droping duplicates records
print(f"duplicate rec :{df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

X = df.drop('labels', axis = 1)
Y = df['labels']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# dimensionality reduction on the training and testing sets
pca = PCA(n_components= 10, random_state=42).fit(X_train)
print(f"The data has been reduced from {X_train.shape[1]} features to -> {len(pca.components_)} features")
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
#print(X_train)
#print(X_test)

# training using Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

#  prediction
Y_pred = rf.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)

print("\nAccuracy using Random Forest Classifier:", accuracy)

print("\n--------------performance metrics--------------")

# classification report showing precision, recall, F1-score, and support
print(classification_report(Y_test, Y_pred))
