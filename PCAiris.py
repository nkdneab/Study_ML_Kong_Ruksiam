from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.decomposition import PCA

#load data
iris = sb.load_dataset("iris")
x = iris.drop("species",axis=1)
y = iris['species']

#pca
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x) #เป็น array

#จับ PAC ยัดเข้า x
#show befor after              #เป็น DF
x['PAC1'] = x_pca[:,0]#ไม่เอาเเถวเอาเเต่คอลัม ที่0
x['PAC2'] = x_pca[:,1]
x['PAC3'] = x_pca[:,2]

x_train,x_test,y_train,y_test = train_test_split(x,y)

#COMPLETEDATA
x_train = x_train.loc[:,['PAC1','PAC2','PAC3']]
x_test = x_test.loc[:,['PAC1','PAC2','PAC3']]

model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

#accuracy score ปกติเเบบไม่ใช้ PCA =95%
print('Accuracy = ',accuracy_score(y_test,y_pred))
