import numpy as np
import math
import matplotlib.pyplot as plt
import operator
# A_x1=np.random.normal(10,3,10)
# print(A_x1)                      #一维正态分布(均值，标准差，长度)

labelsTag=['A','B','C']
testPoint=[20,22]
K=10
def createData():
    #多维正态分布-(均值-N长数组，协方差矩阵-N*N矩阵,数据点个数-size)
    #数据维数绘自动计算，所以这里数据大小是size*N
    #A类数据坐标x,y均值分别为2,3,B类为20,30
    A=np.random.multivariate_normal(mean=[20,3],cov=[[15,0],[0,16]],size=50)
    B=np.random.multivariate_normal(mean=[5,30],cov=[[15,0],[0,16]],size=50)
    C=np.random.multivariate_normal(mean=[35,30],cov=[[15,0],[0,16]],size=50)

    DataSet=np.vstack((A,B,C))    #将两类数据拼接为总数据,竖向拼接

    labels=np.hstack(([labelsTag[0]]*50,[labelsTag[1]]*50,[labelsTag[2]]*50))    #“*50"”对最外层[]里的内容进行重复，再组成一个list
    return DataSet,labels

def drawData(dataSet,labels):
    clusters=[]
    for tag in labelsTag:
        ind=(labels==tag)
        line=plt.scatter(dataSet[ind, 0], dataSet[ind, 1])
        clusters.append(line)
    plt.legend(clusters,labelsTag,loc = 'upper right')


    #没有图例的画法
    # colorMap = np.ones([len(labels)], dtype=np.int16)
    # for ind, c in enumerate(labelsTag):  # 加一个enumerate将list变成迭代对象，便利时同时可以得到下标和list内容
    #     # print(type(ind), c)
    #     colorMap[labels == c] = ind
    # plt.scatter(dataSet[:, 0], dataSet[:, 1], c=colorMap)



def getClass(dataSet,labels,testPoint,k):
    # testPoint=np.tile(testPoint,[dataSet.shape[0],1])    #将testPoint竖向复制[Data.shape[0]次，横向复制1次
    dist=(testPoint-dataSet)**2
    dist=np.sqrt(np.sum(dist,1))     #testPoint-1*2,  Data-150*2   ,没有上面的np.tile语句，这里使用python的广播也可以运行
    argDist=np.argsort(dist)
    # print(argDist)

    #统计前k个最近的点的标签
    countClass={}
    for i in range(k):
        ind=argDist[i]
        pointx=[testPoint[0],dataSet[ind,0]]
        pointy = [testPoint[1], dataSet[ind, 1]]
        plt.plot(pointx,pointy)

        curLabel=labels[ind]
        countClass[curLabel]=countClass.get(curLabel,0)+1
    # print(countClass)
    #dict.items()将dict变成元组，eg: {'A': 1, 'C': 5, 'B': 4}  --->   [('A', 1), ('C', 5), ('B', 4)]
    #sorted可以对元组排序，key参数赋值是一个函数，而operator.itemgetter(1)表明按照依次取每个元组的第1个元素
    #python3的sorted函数取消了cmp参数
    sortedClassCounter=sorted(countClass.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCounter[0][0]   #排序好后sortedClassCounter[0]就是第一个元组就是原dict中值最大的键值对，再进行[0]就是取得键，这里即为标签


if __name__ == '__main__':

    dataSet,labels=createData()

    drawData(dataSet,labels)

    plt.scatter(testPoint[0],testPoint[1])
    testClass=getClass(dataSet,labels,testPoint,K)

    print("测试点(%s,%s)类别为:%s"%(testPoint[0],testPoint[1],testClass))
    plt.show()