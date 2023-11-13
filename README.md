# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and matplot libraries,numpy.
2. import Kmeans algorithm to solve customer segmentation.
3. Using the for loop cluster the given data
4. Predict the output and plot data graphs. 5.Display the outputs

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
data=pd.read_csv("/content/Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)
print('Elbow method diagram:')
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

print('KMeans():')
km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
print('array():')
y_pred=km.predict(data.iloc[:,3:])
y_pred

print('Customer segments diagram:')
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer segments")

```
## Output:
![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/a9790e54-b6cf-4675-80f1-a4708bd81ee5)

![282239446-03863b5b-5592-411f-a91f-0c4a589b8a08](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/55b24195-c801-489d-99bb-f92b9567283b)

![282239447-15e1ecfb-7633-495a-b4e1-66898dbd4dcb](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/1bfc3b22-64af-47b3-b5a8-558e8c327234)

![282239449-8820f2cf-e24f-442d-bc91-489f12f55fce](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/c0d1ff34-d53d-4df5-a5fc-1b5f48a8e4be)

![282239450-5c5ed508-7c6b-428d-a956-5093ef6c4f45](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/24b46a64-efb5-44b4-afd9-f7afca0fbe6b)

![282239450-5c5ed508-7c6b-428d-a956-5093ef6c4f45](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/9cb350cb-d865-4263-94df-0bd00790e200)

![282239451-387d538f-12d3-47da-8b52-2aa0961fef3c](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118703522/1a21289f-f8f6-4aa8-b5c6-891dfe8debc3)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
