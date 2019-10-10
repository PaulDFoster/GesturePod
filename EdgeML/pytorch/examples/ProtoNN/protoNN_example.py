#!/usr/bin/env python
# coding: utf-8

# In[99]:


from __future__ import print_function
import sys
import os
import numpy as np
import torch
import pandas as pd

from pytorch_edgeml.graph.protoNN import ProtoNN
from pytorch_edgeml.trainer.protoNNTrainer import ProtoNNTrainer
import pytorch_edgeml.utils as utils
import helpermethods as helper


# ## Paul's Data
# It is assumed that the USPS data has already been downloaded and set up with the help of `fetch_usps.py` and is placed in the `./usps10` subdirectory.

# In[106]:


# Load data
DATA_DIR = './mpu'
#train, test = np.load(DATA_DIR + '/train.npy'), np.load(DATA_DIR + '/test.npy')
train = np.genfromtxt(DATA_DIR + '/_train.csv', delimiter=",", skip_header=1)
test = np.genfromtxt(DATA_DIR + '/_test.csv', delimiter=",", skip_header=1)
x_train, y_train = train[:, 1:], train[:, 0]
x_test, y_test = test[:, 1:], test[:, 0]

numClasses = max(y_train) - min(y_train) + 1
numClasses = max(numClasses, max(y_test) - min(y_test) + 2) # +2 to allow for zero index
numClasses = int(numClasses)

y_train = helper.to_onehot(y_train, numClasses)
y_test = helper.to_onehot(y_test, numClasses)

print(x_train.shape)
print(y_train.shape)
# Load data
train = np.genfromtxt(DATA_DIR + '/_train.csv', delimiter=",", skip_header=1)
test = np.genfromtxt(DATA_DIR + '/_test.csv', delimiter=",", skip_header=1)
x_train, y_train = train[:, 1:], train[:, 0]
x_test, y_test = test[:, 1:], test[:, 0]
# Convert y to one-hot
minval = 0
#numClasses = int(10)
y_train = helper.to_onehot(y_train, numClasses, minlabel=minval)
y_test = helper.to_onehot(y_test, numClasses, minlabel=minval)
dataDimension = x_train.shape[1]
print(x_train.shape)
print(y_train.shape)


dataDimension = x_train.shape[1]
numClasses = y_train.shape[1]


# ## Model Parameters
# 
# Note that ProtoNN is very sensitive to the value of the hyperparameter $\gamma$, here stored in valiable GAMMA. If GAMMA is set to None, median heuristic will be used to estimate a good value of $\gamma$ through the helper.getGamma() method. This method also returns the corresponding W and B matrices which should be used to initialize ProtoNN (as is done here).

# In[89]:


PROJECTION_DIM = 10
NUM_PROTOTYPES = 20
REG_W = 0.0
REG_B = 0.0
REG_Z = 0.0
SPAR_W = 1.0
SPAR_B = 1.0
SPAR_Z = 1.0
LEARNING_RATE = 0.01
NUM_EPOCHS = 200
BATCH_SIZE = 64
GAMMA = 0.0014


# In[90]:


W, B, gamma = helper.getGamma(GAMMA, PROJECTION_DIM, dataDimension,
                       NUM_PROTOTYPES, x_train)


# In[91]:


protoNNObj = ProtoNN(dataDimension, PROJECTION_DIM, NUM_PROTOTYPES, numClasses,
                     gamma, W=W, B=B)
protoNNTrainer = ProtoNNTrainer(protoNNObj, REG_W, REG_B, REG_Z, SPAR_W, SPAR_B, SPAR_W,
                                LEARNING_RATE, lossType='xentropy')


# In[92]:


protoNNTrainer.train(BATCH_SIZE, NUM_EPOCHS, x_train, x_test, y_train, y_test, printStep=600, valStep=10)


# ## Evaluation

# In[93]:


x_, y_= torch.Tensor(x_test), torch.Tensor(y_test)
logits = protoNNObj.forward(x_)
_, predictions = torch.max(logits, dim=1)
_, target = torch.max(y_, dim=1)
acc, count = protoNNTrainer.accuracy(predictions, target)
W, B, Z, gamma  = protoNNObj.getModelMatrices()
matrixList = [W, B, Z]
matrixList = [x.detach().numpy() for x in matrixList]
sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
nnz, size, sparse = helper.getModelSize(matrixList, sparcityList)
print("Final test accuracy", acc)
print("Model size constraint (Bytes): ", size)
print("Number of non-zeros: ", nnz)
nnz, size, sparse = helper.getModelSize(matrixList, sparcityList,
                                       expected=False)
print("Actual model size: ", size)
print("Actual non-zeros: ", nnz)


# In[94]:


W = W.detach().numpy()
B = B.detach().numpy()
Z = Z.detach().numpy()
W = np.transpose(W)
B = np.transpose(B)
Z = np.transpose(Z)
print(W.shape)
print(B.shape)
print(Z.shape)
np.savetxt("W", W, fmt="%f", delimiter=",")
np.savetxt("B", B, fmt="%f", delimiter=",")
np.savetxt("Z", Z, fmt="%f", delimiter=",")


# In[95]:


gamma


# In[ ]:




