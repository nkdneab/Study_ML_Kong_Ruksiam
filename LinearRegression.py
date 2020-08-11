import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def one():
    x = np.linspace(-5, 5, 100)
    y = 2 * x + 1
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.title("Graph")
    plt.grid()
    plt.show()
def two():
    rng = np.random
    x=rng.rand(50)*10
    y=2*x+rng.randn(50)
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
def model():
    #การจำลองข้อมูล
    rng = np.random
    x = rng.rand(50) * 10
    y = 2 * x + rng.randn(50)
    #linear regression model
    model= LinearRegression()

    #train    อ่านเฉพาะ array 2 มิติ
    xnew=x.reshape(-1, 1)
    model.fit(xnew,y)

    #ตรวจวัดความถูกต้อง
    intercept=model.intercept_ #ค่า C
    coefficient=model.coef_ #ความชัน m
    R_square=model.score(xnew,y)#ตัวสถิติที่ใช้วัดว่าตัวแบบคณิตศาสตร์ที่ได้นี้มีความสมรูปกับข้อมูลมากน้อยอย่างไร หรือรู้จักกัน ในอีกความหมายหนึ่งว่าเป็น ค่าสัมประสิทธิ์แสดงการตัดสินใจ

    #test model
    xfit=np.linspace(-1,11)
    xfit_new=xfit.reshape(-1,1)

    yfit=model.predict(xfit_new)



    #analysis model
    plt.scatter(x,y)
    plt.plot(xfit,yfit)
    plt.show()

model()
