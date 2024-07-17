#step 1 -> Import Libraries
import pandas as pd

# step 2 -> Dataset and informations
data = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.columns)

# step 3 -> define X and y
y = data['diabetes']
X = data[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]

# step 4 -> split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 0.7 ,random_state= 2529)

# step 5 -> select a model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 500)

# step 6 -> train model
model.fit(X_train,y_train)

# step 7 -> predict
y_pred = model.predict(X_test)
print(y_pred)
print(X_test.head())
print(y_test.head())

# step 8 -> accuracy
from sklearn.metrics import accuracy_score
acs = accuracy_score(y_test,y_pred)
print(acs)
