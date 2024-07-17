import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

data = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv')
print(data.head(32))
print(data.tail())

print(data.columns)
print(data.info())
print(data.describe())

y = data['MEDV']
'''
X = [[]]
for i in data.columns:
    if(i not in y):
        X.append(i)'''

# Input features (X)
#X = [col for col in data.columns if col not in y]

#X = data.drop(['MEDV'],axis = 1)
X = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
#print("Target variable (y):", y)
#print("Input features (X):\n", X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
print("X test \n",X_test.head())
print("Y test \n",y_test.head())

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
model = LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)

y_pred = model.predict(X_test)
print(y_pred[:6])

print(mean_absolute_error(y_test,y_pred))
print(mean_absolute_percentage_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))




