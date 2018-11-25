import numpy as np
import math
import operator
#计算数据集的香农熵，数据集M*(N+1),其中M为样本数，N为样本特征数，每行最后一个为标签
def calcuEntropy(DataSet):
    M=len(DataSet)
    LabelCounter={}
    for i in range(M):
        sample=DataSet[i]
        currentLabel=sample[-1]    #每个样本标签
        LabelCounter[currentLabel]=LabelCounter.get(currentLabel,0)+1

    entropy=0.0
    for key,value in LabelCounter.items():    #以键值对的形式遍历dict,每次返回键和对应的值
        prop=value/M
        entropy+=-(prop*math.log(prop,2))    #math.log默认e为底，所以要用第二个参数手动指定底

    return entropy


def createDataSet():
    dataSet=[[1,1,"Yes"],
             [1,1,"Yes"],
             [1,0,"No"],
             [0,1,"No"],
             [0,1,"No"]]
    labels=["no surfing","filppers"]
    return dataSet,labels

#提取样本集dataSet中，第fearureId个特征的值为featureVal的子样本集(子样本集每个样本不再包括第fearureId个样本)
def splitDataSet(dataSet,fearureId,featureVal):
    retDataSet=[]       #因为python的传递方式是引用传递，所以这里重新产生一个自样本集，保证原样本集不被影响
    for curSample in dataSet:
        if curSample[fearureId]==featureVal:
            reducedS=curSample[:fearureId]             #如果当前样本第fearureId个特征的值为featureVal，那么就将该样本
            reducedS.extend(curSample[fearureId+1:])   #除了第fearureId个特征外所有特征代表该样本进行后续的数据集划分
            retDataSet.append(reducedS)     #append将参数看成一个元素直接添加到被添加集合中，无论参数是什么类型都看成一个元素
    return retDataSet                       #extend的参数必须是可迭代对象，其将参数代表的集合中拆开，然后将其中的所有元素依次添加到被添加集合中

#选择数据集的最好划分属性(数据集同样是M*(N+1),N为当前特征数)
def choseBestFeatureToSplie(dataSet):
    M=len(dataSet)
    if M==0:        #要划分的原数据集为空，则返回-1
        return -1
    N=len(dataSet[0])-1     #特征数

    bestFeatureId=-1
    bestInfoGain=0.0
    baseEntropy=calcuEntropy(dataSet)

    for featureId in range(N):     #对于每个特征分别进行计算
        curFeatureList=[sample[featureId] for sample in dataSet]
        curFeatureSet=set(curFeatureList)     #当前特征的所有不重复值
        curEntropy=0.0        #存放以第featureId个特征为划分特征的划分后的新香农熵
        for curFeatureVal in curFeatureSet:
            subDataSet=splitDataSet(dataSet,featureId,curFeatureVal)   #提取子样本集，该子样本集的第featureId个特征的值为curFeatureVal
            prop=len(subDataSet)/float(M)
            curEntropy+=prop*calcuEntropy(subDataSet)   #子样本集信息熵的加权平均

        curInfoGain=baseEntropy-curEntropy
        if curInfoGain>bestInfoGain:
            bestFeatureId=featureId
            bestInfoGain=curInfoGain
    return bestFeatureId

#根据类列表投票选出当前叶子节点代表的类标签
def majorityCnt(classList):
    classCounter={}
    for curClass in classList:
        classCounter[curClass]=classCounter.get(curClass,0)+1
    sortedCounter=sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
    return sortedCounter[0][0]

def createTree(dataSet,labels):
    classList=[sample[-1] for sample in dataSet]   #获得当前数据集的所有标签构成的列表
    if classList.count(classList[0])==len(classList):  #如果当前样本集所有样本都是同一类，那么停止划分，直接返回标签构成决策树的叶子节点
        return classList[0]
    if len(dataSet[0])==1:  #如果当前样本集只剩标签，说明划分到这里，所有的特征都用玩，那么使用投票的方式使用最多类别数作为返回的叶子标签
        return majorityCnt(classList)
    bestFeatureId=choseBestFeatureToSplie(dataSet)   #选择划分属性的Id，这个Id不是全局的，而是近针对当前dataSet的
    bestLabel=labels[bestFeatureId]          #划分属性
    myTree={bestLabel:{}}        #使用划分属性划分当前样本集,用字典表示树
    del(labels[bestFeatureId])   #将当前标签从标签集删除，因为后面的特征Id都是与当前删除后的特征集相对应的
    featureVallist=[sample[bestFeatureId] for sample in dataSet]
    featureValSet=set(featureVallist)
    for val in featureValSet:
        subLabels=labels[:]   #因为之前有del(labels[bestFeatureId])并且python中对于list的传参，是引用传递，之后函数内对于labels
                              #的修改会影响传参前的labels，所以这里复制当前的标签列表然后进行传递
        subDataSet=splitDataSet(dataSet,bestFeatureId,val)  #划分特征的一个值对应一个子树分支
        myTree[bestLabel][val]=createTree(subDataSet,subLabels)
    return myTree



if __name__=="__main__":
    dataSet,labels=createDataSet()
    # entopy=calcuEntropy(dataSet)
    # print(splitDataSet(dataSet,0,1))
    # print(choseBestFeatureToSplie(dataSet))

    myTree=createTree(dataSet,labels)
    print(myTree)
    #{'no surfing': {0: 'No', 1: {'filppers': {0: 'No', 1: 'Yes'}}}}




