# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset. 
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Krithiga U
RegisterNumber:  212223240076
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
HEAD() & INFO() & NULL():

![image](https://github.com/user-attachments/assets/5480d2eb-7f7b-434a-b74f-aff2e2917629)

![image](https://github.com/user-attachments/assets/32ae86e5-5168-49ce-acbf-72232f8b26fa)

![image](https://github.com/user-attachments/assets/468ba664-b7f0-4d9c-ae66-27599504fa69)

data.value_counts()

![image](https://github.com/user-attachments/assets/804f3fa7-bd8e-4322-bce0-ce9edf1bb4fd)

x.head()

![image](https://github.com/user-attachments/assets/f17257b3-4708-4d7f-bab0-2cc755df785c)

accuracy:

![image](https://github.com/user-attachments/assets/d7936122-9dc6-47e6-bfc5-e0005dcaae1b)

Prediction:

![image](https://github.com/user-attachments/assets/39a42532-b090-45c1-be2f-d7c6511be96d)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
