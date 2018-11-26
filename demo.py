import numpy as np
import tensorflow as tf

# arr=np.random.randint(1,10,[4,3])
#
# print(arr)
#
# print(np.mean(arr[1,[1,2]]))
arr=np.zeros((3,2))
arr[0][0]=1
arr[1]=2
arr[2]=3
print(arr)
print("---------------")
# print(2-np.NaN)

var1=tf.Variable(tf.constant([[1,2],[3,4],[5,6]]))
var2=tf.Variable(tf.constant([[1,2],[3,4],[5,6],[7,8]]))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.matmul(var1,var2,transpose_b=True)))

print("---------------")
a=[1,2,3,4,5,6,7]
print(a[-1:-4:-1])   #倒序

print("---------------")
a=[1,2,3]
b=a
b[1]=1000           #list引用传递
print(a)