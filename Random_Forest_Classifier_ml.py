# Importing the necessary libraries
import pandas as pd
import numpy as np

# Loading the dataset from a CSV file into a pandas DataFrame called 'd1'
d1 = pd.read_csv("/content/dataset_analysis1.csv")

pd.set_option("display.max_rows",None)

# Checking for missing values in the DataFrame and display the result
d1.isna().any()

# Drop rows 4963 to 5000 from the DataFrame to remove unwanted data
d2 = d1.drop(d1.index[4963:5000])

# Getting the unique values in the 'Category' and 'Description'column of the DataFrame
d2_unique = d2["Category"].unique()

d2_unique1 = d2["Category"].value_counts()

d3_unique = d2["Description"].unique()

d3_unique1 = d2["Description"].value_counts()

# Importing the necessary encoders from scikit-learn's preprocessing module
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Create an instance of the OneHotEncoder with specified parameters
oneenc = OneHotEncoder(handle_unknown='ignore', sparse_output = False).set_output(transform = "pandas")

# Fit the OneHotEncoder to the 'Category' column and transform it
oneenc_transform = oneenc.fit_transform(d2[["Category"]])

ordenc = OrdinalEncoder(categories = [d3_unique]).set_output(transform = "pandas")


d2['Description'] = ordenc.fit_transform(d2[['Description']])

# Replace the 'Category' column in the DataFrame with the transformed values
d2["Category"] = oneenc_transform["Category_Accessories"]

# Splitting the DataFrame into feature (X) and target (Y) variables
X = d2.iloc[:,4]
Y = d2.iloc[:,1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

# Import the RandomForestClassifier from scikit-learn's ensemble module
from sklearn.ensemble import RandomForestClassifier
Model_project = RandomForestClassifier()

# Reshape the training data for the model
X_train = np.array(X_train).reshape(-1, 1)
Y_train = np.array(Y_train).ravel()
Model_project.fit(X_train, Y_train)

# Reshape the testing data for the model
X_test = np.array(X_test).reshape(-1, 1)
Y_test = np.array(Y_test).ravel()
y_pred = Model_project.predict(X_test)

# Evaluate the performance of the model on the testing data
Model_project.score(X_test,Y_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))

# Creating another instance of the RandomForestClassifier with specified parameters
Model_project2 = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',min_samples_split = 10, max_depth = 14, random_state = 42)

Model_project2.fit(X_train,Y_train)

# Evaluate the performance of the second model on the testing data
y_pred2 = Model_project2.predict(X_test)

Model_project2.score(X_test,Y_test)

Model_project2.get_params()

# Print the classification report for the second model
print(classification_report(Y_test,y_pred2))

