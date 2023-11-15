# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: BASKARAN V
RegisterNumber:  212222230020
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
![239682377-b82352d4-d32b-417f-b084-9bcc6679d6d4](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/4a396f1c-b195-46aa-8564-e4c9aa9e089c)

![239682386-f86626d8-e411-4d23-b827-c9be33f3f986](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/73cde66c-51e8-40bf-9921-1a545645d798)

![239682392-9f764145-534a-4aaa-8573-d9a98b255e71](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/7e97df3f-545e-4477-8757-d8c128f54d6f)
### MSE value
![239682475-ce29115f-52b7-417d-857e-1309409ddd1f](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/31118c43-229f-4644-8578-ba3c155ef315)

### r2 value
![239682499-7cf9ec95-7b09-4250-9a50-09b12bd94308](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/8322f1df-8954-4afe-82ab-a078e15eb24a)

### data prediction
![239682529-cdd07dd7-6bfb-494f-9fe3-378d69126a31](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/a5d4554c-5c12-4868-a0d7-22ecd314f8de)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
