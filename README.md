# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: JANANI R
RegisterNumber: 212224040126 
*/

import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\Downloads\\spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print("Accuracy",acc,end="\n\n")

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
## data
<img width="1048" height="657" alt="image" src="https://github.com/user-attachments/assets/9d3974ed-b5d6-47c5-a2de-68d0a826facc" />

## Confusion matrix
<img width="149" height="59" alt="image" src="https://github.com/user-attachments/assets/eacb273a-7646-430b-8aa1-ff4a47673cdb" />

## accuracy
<img width="310" height="38" alt="image" src="https://github.com/user-attachments/assets/5144122f-e18e-4747-b152-f5a45ab90bcd" />

## classification report
<img width="655" height="199" alt="image" src="https://github.com/user-attachments/assets/518e3549-92dc-414e-aa02-254bdc03831e" />




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
