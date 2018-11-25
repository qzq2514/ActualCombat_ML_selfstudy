import numpy as np
import matplotlib.pyplot as plt
import operator


K=3
testR=0.1
def file2Mat(filepath):
    fr = open(filepath)
    lines = fr.readlines()  # 每一行构成一个元素(包括行内的换行符和制表符)
    numberOfLines = len(lines)
    returnMat = np.zeros([numberOfLines, 3])  # 一行为一个样本，每个样本三个特征
    labels = []
    for ind, line in enumerate(lines):
        line = line.strip()  # 读取每一行，strip()删除字符串首尾指定的字符字符串(默认换行符和空格符)
        featureList=line.split('\t')       #三个特征以空格分开，所以这里用split将特征分开，放在一个数组中
        returnMat[ind, :] = featureList[0:3]  # 文件中每行前三个为特征，最后一个为标签
        labels.append(int(featureList[-1]))
    return returnMat, labels

def drawDataWithLabel(DataSet,Labels):
    labelTagSet=set(Labels)
    Labels=np.array(Labels)
    lines=[]
    for labTag in labelTagSet:
        ind =(Labels==labTag)           #想使用这种找到元素等于指定元素的索引，那么Labels必须是np.array类型，不能是python的list
        line=plt.scatter( DataSet[ind,1],DataSet[ind,2],s=5)
        lines.append(line)
    plt.legend(lines,labelTagSet,loc="upper right")

    # 没有图例形式的直接画三点图，根据标签指定不同类的颜色
    # plt.scatter(DataSet[:,1],DataSet[:,2],c=Labels)

def Normalize(dataSet):
    min=np.min(dataSet,axis=0)       #每个维度的最小值最大值,(3,)）
    max = dataSet.max(axis=0)        #同样可以使用这种方式求跨行的最大值
    normData=np.zeros(dataSet.shape)
    normData=(dataSet-min)/(max-min)         #可以使用np.tile(dataSet.shape[0],1)将其变成与dataSet.shape等大小的N*3数据，然后与da#taSet.shape操作
    return normData,min,max                 #也可以因为min,max都是(3,),dataSet是(N,3)可以使用python的广播

def getClass(normPoint,dataSet,labels,k):
    dist=(normPoint-dataSet)**2
    dist=np.sqrt(np.sum(dist,axis=1))
    argDist=np.argsort(dist)

    k_arg=argDist[0:k]
    labels=np.array(labels)
    kLabels=labels[k_arg]    #取前k个距离最近的点的标签
    # print("前%s个距离最近点的标签:%s"%(k,kLabels))

    labelSet=set(labels)
    countDict={}             #统计各种标签在前k个距离最近的点的标签中出现的次数
    for tag in labelSet:
        countDict[tag]=np.sum(kLabels==tag)
        # print(countDict[tag])

    sortedConter=sorted(countDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedConter[0][0]

def testCalssifier(normData,labels,testRatio,k):
    m=normData.shape[0]    #总样本数
    numTest=int(testRatio*m)    #测试样本数
    correctCount=0
    for i in range(numTest):
        testClass=getClass(normData[i,:],normData[numTest:m,:],labels[numTest:m],k)
        # print("第%s个测试样本被预测为: %s , 其真实类为: %s"%(i+1,testClass,labels[i]))
        correctCount=correctCount+(testClass==labels[i])       #bool类型可以直接和整型运算

    print("该K-Means分类器精度为:%s%%" % ((100*correctCount)/float(numTest)))

def kMeansTest(DataMat,labels,min,max):

    # dange单个样本测试K_means
    testPoint = [18563, 6.93741, 1.3823]
    plt.scatter(testPoint[1], testPoint[2], s=40, c=50)
    normPoint = (testPoint-min) / (max - min)  # 逐位或者广播形式，可以使用list和np.array操作,这里讲测试点归一化
    testCalss = getClass(normPoint, DataMat, labels, K)
    print("最终测试点的分类标签为:", testCalss)
    plt.show()           #展示所有点分布和测试点

    # 测试K-Means分类器精度
    print("精度测试开始........")
    testCalssifier(normData, labels, testR, K)


def DatingTest(normData,labels, min, max):
     resultList=["not at all","in small doses","in large doses"]
     time =float(input("输入查询用户的每周玩游戏时间(h):"))
     salary = float(input("输入查询用户每个月的工资(元):"))
     percentage= float(input("输入查询用户每个月的每天健身时间(h):"))

     dateObject=np.array([salary,time,percentage])
     normPoint = (dateObject - min) / (max - min)
     objClass=getClass(normPoint,normData,labels,K)
     print("约会对象类别:",resultList[objClass-1])

if __name__=="__main__":
    returnMat, labels = file2Mat("DatingData.txt")
    drawDataWithLabel(returnMat, labels)  # returnMat是np.array类型，labels是list类型
    normData, min, max = Normalize(returnMat)


    # kMeansTest(normData,labels, min, max)      #简单的k-Means测试
    DatingTest(normData,labels, min, max)        #约会测试


