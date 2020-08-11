import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")

# train and test set
x= dataset["MinTemp"].values.reshape(-1,1)
y= dataset["MaxTemp"].values.reshape(-1,1)


# 80% and 20%

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# training
model = LinearRegression()
model.fit(x_train,y_train)
#print(model.score(x_train,y_train))

#test
y_pred = model.predict(x_test)

#analysis model
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color="red",linewidth=2)
plt.show()

#compare real data and predict data
df = pd.DataFrame({'Actually':y_test.flatten(),'Predict':y_pred.flatten()})


df1 = df.head(20)
df1.plot(kind='bar',figsize=(16,10))
plt.show()

#เเสดงผลเเบบเปรียบเทียบค่าด้วย MAE MSE RMSE RScore
print("MAE =",metrics.mean_absolute_error(y_test,y_pred))
print("MSE =",metrics.mean_squared_error(y_test,y_pred))
print("RMSE =",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Score =",metrics.r2_score(y_test,y_pred))