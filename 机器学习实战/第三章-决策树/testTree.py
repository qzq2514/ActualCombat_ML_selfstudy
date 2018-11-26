import treeModel
import treePlot

dataSet=[]
featureList=["age","pres","asti","tear"]    #各个特征名称
with open("testData.txt") as fr:
    lines=fr.readlines()
    for line in lines:
        sample=line.strip().split("\t")
        # print(type(sample))
        dataSet.append(sample)


# print(dataSet)

# myTree=treeModel.createTree(dataSet,featureList)
# treeModel.saveTree(myTree,"qzq_tree.txt")

loadT=treeModel.loadTree("qzq_tree.txt")
print(loadT)
# treePlot.createPlot(myTree)