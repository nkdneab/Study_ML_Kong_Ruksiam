from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("diabetes.csv")

x = df.drop('Outcome',axis=1)#axis=0 คือเเถว

#เอาคอลลัมออกให้พร้อมใช้ #ค่าต่างๆ
x=x.values
#ผลเฉลย
y=df['Outcome']

#training and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

#train
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)

#predict
pred=knn.predict(x_test)

print(pd.crosstab(y_test,pred,rownames=["Actually"],colnames=['Prediction'],margins=True))