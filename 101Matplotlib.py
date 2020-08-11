import matplotlib.pyplot as plt
import numpy as np
#ตอนที่1
#plt.plot([28,29,29,29])
#plt.show()

#plt.plot([1,2,3,4],[28,29,29,29])
#plt.show()

#plt.plot([1,2,3,4],[28,29,29,29],'o')
#plt.show()

#plt.plot([1,2,3,4],[28,29,29,29],'ro')
#plt.plot([1,2,3,4],[2,3,4,9],'go-')
#plt.plot([1,2,3,4],[6,9,2,2],'k')
#plt.show()
#x1,y1=[1,2,3,4],[28,29,29,29]
#x2,y2=[1,2,3,4],[2,3,4,9]
#x3,y3=[1,2,3,4],[6,9,2,2]

#plt.plot(x1,y1,color='pink',marker='o')
#plt.plot(x2,y2,color='green',linestyle='--')
#plt.plot(x3,y3,color='blue',linewidth=4)
#plt.show()

######
#x = range(2014,2021)
#y = [20,32,23,43,21,22,19]
#plt.plot(x[:4],y[:4],color='deepskyblue',label='actual')
#plt.plot(x[3:],y[3:],color='deepskyblue',linestyle='--',label='prediction')
#plt.title('sales prediction')
#plt.ylabel("sales ('000)")
#plt.ylabel("year")
#plt.legend()
#plt.show()
######
#x1,y1=[1,2,3,4],[28,29,29,29]
#avg=sum(y1)/4
#ytick =('mocha','latte','espresso','tea')
#plt.barh(x1,y1)
#plt.axvline(avg,color='red',linestyle='--')
#plt.title('Orders by manu\nFeb 2017',color='orange', fontsize=17)
#plt.xlabel("# orders")
#plt.yticks(x1,ytick)
#plt.show()

#x1,y1=[1,2,3,4],[28,29,29,29]
#xticks =('mocha','latte','espresso','tea')
#fig,ax =plt.subplots(1,2) #axบอกถึงตัวไหน figบอกถึงพื้นที่
#ax[0].bar(x1,y1,color='green') #ax[0]การบอกถึงตัวเเรก
#ax[1].barh(x1,y1,color='blue')
#plt.sca(ax[0])
#plt.xticks(x1,xticks)
#plt.sca(ax[1])
#plt.yticks(x1,xticks)
#fig.tight_layout() #ทำให้เเบ่งเป็นพื้นที่ๆไม่ทับกัน
#plt.show()

#label = ('mocha','latte','espresso','tea')
#val = (30,40,45,50)
#plt.pie(val,labels=label,startangle=90,autopct='%1.2f%%'
#        ,explode=(0,0,0.1,0)
#        ) #startangle เริ่มต้นที่ 90 องศา autopct ทำเปอรเซน
#plt.axis('equal')#ทำให้วงกลทสมมาตร
#plt.show()

#label = np.array(['mocha','latte','espresso','tea'])
#val = (30,40,45,50)
#explode = np.zeros(label.size) #[0. 0. 0. 0.]
#explode[np.where(label == 'mocha')] = 0.1
#colors=["red","pink",'orange',"green"]
#plt.pie(val,labels=label,startangle=90,autopct='%1.2f%%'
#        ,explode=explode,colors=colors
#        )
#plt.axis('equal')
#plt.show()

labels = np.array(['wisesight','company2','company3','company4'])
x = np.arange(labels.size) #[0 1 2 3 4 5]
y1 = np.random.normal(80,10,x.size)
y2 = np.random.normal(95,12,x.size)
plt.bar(x,y1,color='pink',alpha=0.3,label='Number(Jan-Jan 2019)') #alpha บอกถึงความโปร่งเเสง
plt.plot(x,y2,color='orange',marker='o',label='Average(Dec-Dec 2018)')
plt.xticks(x,labels)
plt.legend()
plt.title('Number of worker(Jan-Jan 2019)')
plt.ylabel("Number of worker('000)")
plt.show()