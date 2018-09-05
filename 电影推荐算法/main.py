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

                                       #公众号上直接用ratings_df["movieRow"].max()+1，我仔细想了下也有道理，如果后面的电影都没有被评论，那么就没必要在
                                       #评分表和记录表中特地为他们预留空间，反正以后进行电影推荐的时候也不会将没有被评分过的电影推荐给用户
                                       #实际上，movieRow为9123,9124都没有被任何用户评论
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
print(rating.shape)



#将评分减去平均值，是为了防止有用户没有对任何电影评论或者有电影没有被任何用户评论
#这样的话，最后会导致该用户的喜好特征或者该电影的类型成分都是0向量，最后进行电影推荐预测时得到也都是0向量，导致一种死循环
#而先减去平均值，最后即便得到0向量，再加上平均值后也不会产生0向量的死循环(详见"算法笔记/7.jpg")
def normalizeRating(_rating,_record):
    m,n=_rating.shape     #m,n分别表示电影数和用户数
    _rating_mean=np.zeros((m,1))    #每部电影的平均评分
    _rating_meaned=np.zeros((m,n))    #

    for i in range(m):        #遍历每部电影(电影下标从0开始)
        idx=record[i,:]!=0       #找到第i部电影被评分过的用户下标
        _rating_mean[i][0]=np.mean(_rating[i,idx])   #求第i电影所获得的平均评分
        _rating_meaned[i,idx]-=_rating_mean[i][0]     #某部电影的所有评分减去该电影的平均分
    return _rating_mean,_rating_meaned   #这里np.mean求出来是一个实数，可以直接赋值给大小是[m,1]的_rating_mean[i]([1,]的矩阵)
                                         #也可以直接赋值给_rating_mean[i][0]（是个实数，按理说应该使用这种赋值，但是python允许把实数赋值给[1,]大小的矩阵）
                                         #如果这里_rating_mean是[m,2],再进行"_rating_mean[i]=kk"的赋值，那么就会采用广播的形式，将_rating_mean第i行全部赋值为kk

rating_mean,rating_meaned=normalizeRating(rating,record)
# print(rating_meaned)

rating_meaned=np.nan_to_num(rating_meaned)   #如果有较多的电影没有被评分，那么第i个idx就是空集合，则再进行np.mean还是得到空值
rating_mean = np.nan_to_num(rating_mean)  #这样就会导致最后rating_mean和rating_meaned很多都是空值，使用nan_to_num将空值变为0




