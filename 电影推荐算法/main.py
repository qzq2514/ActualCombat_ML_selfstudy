# from:顶级程序员-基于python和tensorflow的电影推荐算法
# 数据来源:https://grouplens.org/datasets/movielens/


import pandas as pd
import numpy as np
import tensorflow as tf

#第一步.数据清洗和初步处理
#ratings.csv:用户userId对某部电影movieId的评分
ratings_df=pd.read_csv("ml-latest-small/ratings.csv")
# print(ratings_df.tail())

#movies.csv:第movieId部电影的名称(title)和流派(genres)
movies_df=pd.read_csv("ml-latest-small/movies.csv")
# print(movies_df.tail())
movies_df["movieRow"]=movies_df.index   #根据电影表的index生成新列，因为movies.csv中的movieId不是连续的，不便于后买呢操作
# print(movies_df.tail())
# print(len(movies_df["movieRow"]))

#提取原电影csv中的电影id，名称和新列（movieRow）并保存成一个新文件,相当于用movieRow代替movieId
movies_df=movies_df[["movieRow","movieId","title"]]
movies_df.to_csv("ml-latest-small/moviesProcessed.csv",index=False,header=True,encoding="utf-8")
# print(movies_df.head())


##根据movieId关键字对评分表和电影表进行合并，默认how="innder",双方都有的健值才会合并
#movies_df中movieId是不重复的，但是ratings_df中movieId是重复的，即不同的人对同一部电影进行评分
#这时候会将movies_df中对应的movieId行重复拼接到ratings_df中
ratings_df=pd.merge(ratings_df,movies_df,on="movieId")
# print(ratings_df.head())

ratings_df=ratings_df[["userId","movieRow","rating"]]
ratings_df.to_csv("ml-latest-small/ratingProcessed.csv",index=False,header=True,encoding="utf-8")
# print(ratings_df.head())



#第二步.建立评分表和记录表
userNo=ratings_df["userId"].max()+1    #userId:1-671,因为userId是连续的(且从下标1开始)，直接用原来的找最大值可以找到
movieNo=movies_df["movieRow"].max()+1  #原先的movieId不是连续的，所以之前要定义movieRow，以便这里找到总电影部数
                                       #求总电影数，要用movies_df["movieRow"]，而不要用ratings_df["movieRow"]，因为
                                       #此时ratings_df是合并后的表，有可能有的电影从来没被评分过，这时就导致合并前的ratings_df的"movieId"不全
                                       #这时再进行how="inner"的合并，会导致最后合并的表中"movieRow"不会包括所有的电影
                                       #而且这里movieRow是从0-9124

rating=np.zeros((movieNo,userNo))    #表示评分记录表，rating[i,j]表示电影i被用户j评论了额
flag=0
ratings_df_len=len(ratings_df)   #求表的行数，也可以用np.shape(ratings_df)[0]

# print(ratings_df.head())
# print(ratings_df.columns)
for ind,row in ratings_df.iterrows():  #遍历每一行,ind是整数(从0开始)，row是一个Series，键是原DataFrame的columns,值是每一行的值
    mId=int(row["movieRow"])
    uId=int(row["userId"])
    rating[mId,uId]=row["rating"]         #建立评分表
    flag+=1

record=rating>0      #建立记录表，有评分的置为1,保证最后记录表中只有0,1两种值
record=np.array(record,dtype=np.int)   #bool矩阵转为int型
# print(record[2,30])    --False
# print(record)