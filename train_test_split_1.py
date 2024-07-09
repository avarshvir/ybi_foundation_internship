import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
salary = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Salary%20Data.csv')

# Display the first few rows of the dataset
print("Salary Head \n",salary.head())

# Separate the features (X) and target variable (y)
y = salary['Salary']
X = salary[['Experience Years']]

#columns
print("Columns \n",salary.columns)

# Print shapes of the dataset to verify
print("Shape \n",salary.shape)
print("X.shape \n",X.shape)
print("y.shape \n",y.shape)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Print shapes of the split datasets
print("X_train \t, X_test \t, y_train \t, y_test :")
print(X_train.shape, "\t", X_test.shape, "\t", y_train.shape, "\t", y_test.shape)

# Print the training data (X_train) to verify
print(X_train)
# print(X_test) # You can print X_test if needed
