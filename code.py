from mnist.loader import MNIST
import numpy as np
import pandas as pd
import random
from numpy import linalg as LA
from matplotlib import pyplot as plt 

mndata = MNIST('image')

train, labels = mndata.load_training()
num = input("Enter a number")

K = 400


Data = []

for i in range(0,len(train)):
	# print(labels[i])
	if int(labels[i]) == int(num):
		Data.append(train[i])


XY = np.array(Data)

# noise = np.random.normal(50, 50, XY.shape)*0.002*255
# XY = np.add(XY, noise)

Data = list(XY)
mean = []
Y = []
for i in range(0, 784):
	X = []
	for k in range(0, len(Data)):
		X.append(Data[k][i])
	mean.append(np.mean(X))
	Y.append(X)
# print(Y)
cov_mat = np.cov(Y)
EG, EV = LA.eig(cov_mat)

print(len(EG));
print(EG[0]);

print(len(EV));
print(len(EV[0]));

Z = np.dot(Data, EV)
EV = EV.transpose()
Z = np.dot(Z, EV)
for i in range(0, 784):
	# print(i%28, int(i/28))
	plt.scatter(i%28, 28-int(i/28), c = 'gray', s =Data[0][i])
# plt.show()
# # print(EV)
# for i in range(0, 784):
# 	# print(i%28, int(i/28))
# 	plt.scatter(i%28, 28-int(i/28), c = 'gray', s =Z[0][i])
plt.show()


idx = EG.argsort()[::-1]   
EG = EG[idx]
EV = EV[:,idx]
# print(EG)
EVup = np.split(EV.transpose(), [K])

EVup[0] = EVup[0].transpose()

Z = np.dot(Data, EVup[0])
Z = np.dot(Z, EVup[0].transpose())


for i in range(0, 784):
	# print(i%28, int(i/28))
	plt.scatter(i%28, 28-int(i/28), c = 'gray', s =Z[0][i])
plt.show()
