import numpy as np
import matplotlib.pyplot as plt


def ROCPolt(pred,lab):
    print(lab)
    numPosClass=np.sum(lab==1.0)
    numNegClass=lab.shape[0]-numPosClass
    yStep=1/numPosClass
    xStep=1/numNegClass
    # print(numPosClass,numNegClass)
    ySum=0.0        #用于计算AUC(ROC曲线下的面积)
    plt.figure()
    curX=0.0;curY=0.0           #一开始阈值极高，所有样本都被预测为负样本
    sortPred=pred.argsort()[::-1]    #预测得分的从小到大排序
    # print(sortPred)
    for threshInd in sortPred:  #以每个预测为阈值，大于等于该阈值预测为正类
        nextX=curX;nextY=curY
        if lab[threshInd]==1:    #又一个正样本被分类正确，则真正例数目多一个,y坐标变大
            nextY+=yStep
            # ySum+=yStep
        else:                    #又一个负样本被误分类，则假正例数目增加，x坐标变大
            nextX+=xStep
            ySum+=nextY
        print("ySum:",ySum)
        plt.plot([curX,nextX],[curY,nextY],'b')
        curX=nextX;curY=nextY
    plt.plot([0.0,1.0],[0.0,1.0],'b--')
    plt.show()
    return ySum*xStep



predClass=np.array([0.3,0.5,0.48,0.28,0.94,0.85,0.37,0.67,0.12,0.23])
label=np.array([1,0,1,0,1,1,0,1,0,0],np.int8)

auc=ROCPolt(predClass,label)
print("AUC:",auc)     #0.88