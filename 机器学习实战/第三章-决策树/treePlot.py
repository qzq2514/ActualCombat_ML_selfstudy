import matplotlib.pyplot as plt

#定义文本框格式(边框样式和前景灰度值)和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")   #默认箭头是直线，还可以设置connectionstyle="arc3,rad=0.2"使得箭头有弧度
                                   #arrowstyle="<-"是被标注位置指向标注文本位置,arrowstyle="->则反过来"

def plotNode(nodetxt,centerPoint,parentPoint,nodeType):
    #第一个参数nodetxt是文本内容
    #xy:被标注位置,xytext:标注文本位置，
    #xycoords和textcoords表明以上两个位置的给出方式
    #'axes fraction'表明(0,0)是轴域左下角,(1,1)是右上角 还可以使用"data"表示使用轴域数据坐标系
    #bbox是注释文本框样式，arrowprops是箭头样式
    createPlot.ax1.annotate(nodetxt,xy=parentPoint,xycoords='data',
                            xytext=centerPoint,textcoords='axes fraction',
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def plotMidText(sonNode,parentNode,textStr):
    midX = (sonNode[0] + parentNode[0]) / 2
    midY = (sonNode[1] + parentNode[1]) / 2
    createPlot.ax1.text(midX,midY,textStr)



#开始画决策树(dict形式的数,根节点坐标,根节点显示内容)
def plotTree(myTree,rootPoint,nodeText):
    numLeaves=getNumLeaves(myTree)
    depth=getTreeDepth(myTree)
    rootStr=list(myTree.keys())[0]

    #讲真，一波距离变换把我搞懵逼了，这是我直接cv过来的(command+v--->command+v)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeaves)) / plotTree.totalW / 2.0, plotTree.yOff)
    print(rootStr, cntrPt, rootPoint)
    plotMidText(cntrPt, rootPoint, nodeText)
    plotNode(rootStr, cntrPt, rootPoint, decisionNode)
    subTree = myTree[rootStr]

    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in subTree.keys():
        if type(subTree[key]).__name__ == 'dict':
            plotTree(subTree[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(subTree[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor="white")    #plt.figure(窗口编号，窗口前景色)
    fig.clf()     #清除当前窗口中的数字
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)    #python函数内创建的变量默认都是全局的，所以createPlot.ax1在别的函数中也可以使用

    plotTree.totalD=getTreeDepth(inTree)       #获取输的深度和叶节点数(宽度)
    plotTree.totalW = getNumLeaves(inTree)
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff = 1.0

    # plotMidText((0.2,1),(0,5,0,4),"qzq2514")
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

    # plotNode("Decision Node",(0.5,0.1),(0.1,0.5),decisionNode)   #测试用
    # plotNode("Leaf Node",(0.8,0.1),(0.3,0.8),leafNode)

#获得树叶子节点-树以dict形式存放
def getNumLeaves(mytree):
    numLeaves=0         #树都是一个长度为1的dict,且唯一的那个键就是划分属性-根节点
    root=list(mytree.keys())[0] #树的字典形式，最外层是第一个划分属性,也是根节点,mytree.keys()返回dict_keys类型，必须转为list类型才能用下标索引
    subTree=mytree[root]  #得到第一节点下划分出来的子树，也是一个字典形式
    for key in subTree.keys():
        if type(subTree[key]).__name__=="dict":    #判断某个变量是不是某个类型:type(xxx).__name__=="类型名(dict,int,list等)"
            numLeaves+=getNumLeaves(subTree[key])  #如果下一个也是一个dict，说明也是一个子树，那么递归找到子树的叶节点数量
        else:
            numLeaves+=1    #否则遇到叶子节点，加一就行
    return numLeaves

#获得树深度-树以dict形式存放
def getTreeDepth(mytree):
    root=list(mytree.keys())[0]
    subTree = mytree[root]
    maxDepth=0
    for key in subTree.keys():
        curSubTreeDepth=0
        if type(subTree[key]).__name__=="dict":      #找到每棵子树的深度,如果是叶子节点，那么直接深度认定为0
            curSubTreeDepth=getTreeDepth(subTree[key])

        if curSubTreeDepth>maxDepth: #找到最深子树的深度
            maxDepth=curSubTreeDepth
    return maxDepth+1    #最大子树深度加1(如果)


#测试用，里面是一个树的集合，得到第i个树
def retrieveTree(i):
    listOfTree=[{"no surfing":{0:"No",1:{"flippers":{0:"No",1:"Yes"}}}},
                {"no surfing":{0:"No",1:{"flippers":{0:{"head":{0:"No",1:"Yes"}},1:"No"}}}}
                ]
    return listOfTree[i]


if __name__=="__main__":

    testTree=retrieveTree(0)
    print(getNumLeaves(testTree))
    createPlot(testTree)
    # print(getTreeDepth(testTree))


