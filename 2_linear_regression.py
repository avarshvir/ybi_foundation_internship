import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Salary%20Data.csv')
print(data.head(30))

#columns
print(data.columns)

#info
print(data.info())

#describe
print(data.describe())

# Plotting 'Experience Years' vs 'Salary'
plt.figure(figsize=(10, 6))
plt.scatter(data['Experience Years'], data['Salary'], color='blue', alpha=0.5)
plt.title('Experience Years vs Salary')
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

# Separate the features (X) and target variable (y)
y = data['Salary']
X = data[['Experience Years']]

# Print shapes of the dataset to verify
print("Shape \n",data.shape)
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

# Step 5 : select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 6 : train or fit model
model.fit(X_train,y_train)

#intercept and slope
print("intercept :", model.intercept_)
print("slope : ",model.coef_)

# Step 7 : predict model
y_pred = model.predict(X_test)

print(y_pred)
print(X_test)

# Step 8 : model accuracy
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

print(mean_absolute_error(y_test,y_pred))
print(mean_absolute_percentage_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
