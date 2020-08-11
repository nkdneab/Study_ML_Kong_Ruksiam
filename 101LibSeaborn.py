import seaborn as sb
import matplotlib.pyplot as plt
def iris():
    iris_dataset = sb.load_dataset('iris')

    sb.set()
    sb.pairplot(iris_dataset, hue='species', height=2)
    plt.show()
#libที่ดีควรศึกษาเพิ่มเอาไว้ใช้คำนวนเเละสร้างกราฟ
iris()