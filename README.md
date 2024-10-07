# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries. 
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:KRITHIGA U
RegisterNumber:  212223240076
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
##HEAD() & INFO():

![image](https://github.com/user-attachments/assets/704e5cb7-0ded-461e-8724-97a729810f29)


##NULL & COUNT:

![image](https://github.com/user-attachments/assets/76def074-91c4-4ad5-aa13-c568507e5211)
![image](https://github.com/user-attachments/assets/80eefd9a-355f-4489-8445-eabdd154b60d)


##ACCURACY SCORE:

![image](https://github.com/user-attachments/assets/b50cc95a-cbbc-4a67-9c58-5e22717dede7)


##DECISION TREE CLASSIFIER MODEL:

![image](https://github.com/user-attachments/assets/0cd2e0fa-9bf1-49b6-b472-c33f29143b9b)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
