import numpy as np
import matplotlib.pyplot as plt
def getSimpleData():
    dataMat=np.array([[1.,2.1],
             [1.5,1.6],
             [1.3,1.],
             [1.,1.],
             [2.,1.]])
    classLabel=np.array([1.0,1.0,-1.0,-1.0,1.0])
    return dataMat,classLabel

#单特征分类器，数据集dataMat中第dimen维特征与阈值threshVal比较后的样本归为-1类,threshInv决定不等号方向
def SinglefeatureClassify(dataMat,dimen,threshVal,threshInv):
    retLabel=np.ones((dataMat.shape[0],1))   #长度为样本数的标签向量
    if threshInv:
        retLabel[dataMat[:,dimen]<=threshVal]=-1.0
    else:
        retLabel[dataMat[:, dimen] > threshVal] = -1.0
    return retLabel


def getBestFeatureInfo(dataMat, labels, sampleWeight):
    stepNum=11
    bestFeatureInfo={}
    m,n=dataMat.shape
    minError=np.Inf
    bestPred=labels.copy()
    for dimen in range(n):
        featureMin=dataMat[:,dimen].min()      #当前特征的最小值和最大值，用以确定阈值步长
        featureMax= dataMat[:, dimen].max()
        stepSize=(featureMax-featureMin)/stepNum
        for setp in range(-1,int(stepNum)+1):     #阈值在特征的真实范围之外也是可以的，所以这里从-1开始，
            curThresh=featureMin+setp*stepSize
            for invChoice in [True,False]:     #不同的等号方向
                predVal=SinglefeatureClassify(dataMat,dimen,curThresh,invChoice)
                errorArr = np.zeros((m,1))
                errorArr[predVal!=labels]=1     #得到根据当前单特征决策信息构成的分类器进行划分后的预测错集


                weightError=np.sum(errorArr*sampleWeight)   #计算当前单特征分类器的加权误差率,这里errorArr和sampleWeight都是np.array，直接*是点乘
                if weightError<minError:
                    minError=weightError
                    bestPred=predVal.copy()
                    bestFeatureInfo["dimen"]=dimen      #保存当前单特征分类器
                    bestFeatureInfo["thresh"] = curThresh
                    bestFeatureInfo["invChoice"] = invChoice
    return bestFeatureInfo,minError,bestPred

def AdaBoostTrain(dataMat,label,iteraNum=40):
    weakClassifyArray=[]
    m=dataMat.shape[0]
    sampleWeight=np.ones((m,1))/m    #初始化样本权重
    weightClass=np.zeros((m,1))       #当前所有弱分类器的加权组合后对每个样本的分类预测(可能是小数)
    for i in range(iteraNum):
        # print(sampleWeight)
        classsfy, minError, predClass = getBestFeatureInfo(dataMat, label, sampleWeight)    #根据当前样本权重得到这次迭代的弱分类器-classsfy
        alpha=1/2*np.log((1-minError)/(max(minError,1e-10)))     #当前弱分类器在总分类器的权重-精确度越高，权重越大
        classsfy["alpha"]=alpha     #为弱分类器添加权重属性，这与之前划分特征、阈值、符号翻转一起构成单特征弱分类器的特征
        weakClassifyArray.append(classsfy)
        expon=-alpha*(predClass*label)      #开始计算下一次迭代用到的样本权重向量,增大误分类样本的权重
        sampleWeight=sampleWeight*np.exp(expon)
        sampleWeight=sampleWeight/sampleWeight.sum()

        weightClass+=alpha*predClass      #当前弱分类器对于最终预测结果的影响(带有权值)
        # print(weightClass)
        curError=np.sign(weightClass)!=label   #np.sign将连续的预测weightClass变为离散的预测(-1和1类)
        curError=np.array(curError,np.int8)
        errorRate=curError.sum()/m     #当前总分类器的精度
        if errorRate==0.0:
            break
    return weakClassifyArray

data,label=getSimpleData()
label=np.array(label)[:,np.newaxis]

weakClassifyArray=AdaBoostTrain(data,label)

# sampleW=np.ones((data.shape[0],1))/data.shape[0]
# classsfy,minError,newLabel=getBestFeatureInfo(data,label,sampleW)
# print(minError)
# newLabel=SinglefeatureClassify(data,0,1.3,True)
# plt.figure(1)
# plt.scatter(data[:,0],data[:,1],s=60,c=label)
#
# plt.figure(2)
# plt.scatter(data[:,0],data[:,1],s=60,c=newLabel)
# plt.show()
