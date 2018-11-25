import numpy as np
import os
import cv2
import operator

imgPath="digits/trainingDigits/"

def getDataSet(path):
    fileList = os.listdir(path)
    m=len(fileList)
    dataSet=np.zeros([m,32*32])
    labels=[]
    for ind,filename in enumerate(fileList):
        labels.append(int(filename[0]))
        sample=getSample(path+filename)     #得到每个行向量形式的样本
        # print(sample)
        dataSet[ind,:]=sample
    return dataSet,np.array(labels)


def getSample(filePath):
    pic = np.zeros([32, 32], dtype=np.float32)
    with open(filePath) as fr:
        for i in range(32):
           lineStr=fr.readline()
           for j in range(32):
               pic[i,j]=lineStr[j]
    # cv2.imshow("Digits",pic)      #可以显示单张图片
    # cv2.waitKey(0)
    return pic.reshape([32*32])   #展开成行向量形式，因为后面数据集就是以行为样本


def getClass(dataSet,labels,testPic,k):
    dist=(testPic-dataSet)**2
    dist=np.sqrt(np.sum(dist,1))
    argDist=np.argsort(dist)

    #统计前k个最近的点的标签
    countClass={}
    for i in range(k):
        ind=argDist[i]
        curLabel=labels[ind]
        countClass[curLabel]=countClass.get(curLabel,0)+1
    sortedCounter=sorted(countClass.items(),key=operator.itemgetter(1),reverse=True)
    return sortedCounter[0][0]

if __name__=="__main__":
    treainDataSet,trainLabels=getDataSet(imgPath)
    # print(labels)
    testDataSet, testabels = getDataSet(imgPath)
    testM=len(testabels)
    pred=np.ones([testM],dtype=np.int16)
    for i in range(testM):
        predictLabel=getClass(treainDataSet,trainLabels,testDataSet[i,:],10)
        pred[i]=predictLabel
        # print("groundTruth: %s, prediction:%d " %(testabels[i],pred[i]))

    correctCount=np.sum(testabels==pred)
    print("手写识别字符精确度:",float(correctCount)/testM)