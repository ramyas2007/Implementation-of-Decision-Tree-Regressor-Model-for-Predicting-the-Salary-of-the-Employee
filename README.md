# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder. 
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAMYA S
RegisterNumber: 212224040268
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("/content/Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![image](https://github.com/user-attachments/assets/211e7881-c9d6-48f4-97ea-895aca7eebc9)
![image](https://github.com/user-attachments/assets/405bac59-a6b1-4271-81ce-b52951d867b0)
![image](https://github.com/user-attachments/assets/2c069602-2e87-4e1e-a2c0-cc1bc56f5c01)
![image](https://github.com/user-attachments/assets/a5a5b106-a3a4-4890-8ece-1635c2c973eb)
![image](https://github.com/user-attachments/assets/3e674e0d-6b25-41da-ab92-60a417e9fa0c)
![image](https://github.com/user-attachments/assets/74fb9217-376b-42b3-b2d2-7d181745ef8a)
![image](https://github.com/user-attachments/assets/bf6b912f-327d-4a0b-bccf-9b95053cf7a7)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
