from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#สร้างข้อมูลมั่วๆ
x,y=make_blobs(n_samples=100,n_features=10)


print("Before = ",x.shape)
pca = PCA(n_components=4)
x=pca.fit_transform(x)
print("After = ",x.shape)

#ใช้ดูว่ามีกี่ componant
print(pca.n_components_)

#ใช้ดูว่าเเต่ละ componant สามารถนำไปหาความสัมพันได้กี่เปอเซ็น
print(pca.explained_variance_ratio_)

#สร้างกราฟ เเสงดงความสามารถนำไปหาความสัมพันได้กี่เปอเซ็น
df = pd.DataFrame({'Variance Explained':pca.explained_variance_ratio_,'Principal Component':['PC1','PC2','PC3','PC4']})
sb.barplot(x='Principal Component',y='Variance Explained',data=df,color='c')
plt.show()