# Importing necessary libraries
import pandas as pd
import numpy as np

# Loading the dataset from a CSV file
d1 = pd.read_csv("/content/dataset_analysis1.csv")

# Setting the display option to show all rows
pd.set_option("display.max_rows",None)

d1.isna().any()

# Drop unwanted rows (rows 4963 to 5000) from the dataset which is having null 
d2 = d1.drop(d1.index[4963:5000])

# Get unique values and value counts of the "Category" and "Description" column
d2_unique = d2["Category"].unique()

d2_unique1 = d2["Category"].value_counts()

d3_unique = d2["Description"].unique()

d3_unique1 = d2["Description"].value_counts()

# Import necessary modules for encoding categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Create a OneHotEncoder instance to encode the "Category" column
oneenc = OneHotEncoder(handle_unknown='ignore', sparse_output = False).set_output(transform = "pandas")

oneenc_transform = oneenc.fit_transform(d2[["Category"]])


d2["Category"] = oneenc_transform["Category_Accessories"]

# Import seaborn for data visualization
import seaborn as sns

# Create a scatter plot to visualize the relationship between "Category" and "ID" columns
sns.scatterplot(x = "Category", y = "ID", data = d2)

ordenc = OrdinalEncoder(categories = [d3_unique]).set_output(transform = "pandas")

d2['Description'] = ordenc.fit_transform(d2[['Description']])

d2["Category"] = oneenc_transform["Category_Accessories"]

# Select the feature and target columns for modeling
X = d2.iloc[:,4]
Y = d2.iloc[:,1]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

# Create a LogisticRegression instance for modeling
from sklearn.linear_model import LogisticRegression
Model_project = LogisticRegression()

# Fit the model to the training data and reshaping
X_train = np.array(X_train).reshape(-1, 1)
Y_train = np.array(Y_train).ravel()
Model_project.fit(X_train, Y_train)

X_test = np.array(X_test).reshape(-1, 1)
Y_test = np.array(Y_test).ravel()
y_pred = Model_project.predict(X_test)

# Evaluate the model's performance on the testing data
Model_project.score(X_test,Y_test)

# Print the classification report for the model
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))

