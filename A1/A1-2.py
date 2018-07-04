
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

train = pd.read_csv('Dataset_2_train.csv',header=None,usecols=[0,1])
valid = pd.read_csv('Dataset_2_valid.csv',header=None,usecols=[0,1])
test = pd.read_csv('Dataset_2_test.csv',header=None,usecols=[0,1])


# In[53]:


def learn(train,valid,a,threshold):
    ip = np.asarray(train[0])
    op = np.asarray(train[1])
    cfPrev = np.array([1,1],dtype=float)
    cf = np.array([0,0],dtype=float)
    index = np.arange(0,len(ip))
    epoch = 0   
    while (np.linalg.norm(cf - cfPrev)) > threshold:
        epoch+=1
        random.shuffle(index)
        cfPrev[0] = cf[0]
        cfPrev[1] = cf[1]
        for i in range(len(ip)):
            j  = index[i]
            cf[0] = cf[0] - a*(cf[0]+cf[1]*ip[j] - op[j])
            cf[1] = cf[1] - a*(cf[0]+cf[1]*ip[j] - op[j])*ip[j]        
        plt.plot(epoch,mse(cf,valid),'o') 
    plt.xlabel("# Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Learning Curve")
    plt.show()
    print ("Fianl Validation MSE:", mse(cf,valid))
    print ("# Epoch:" , epoch)
    return cf


# In[26]:


def mse(cf, df):
    ip = np.asarray(df[0])
    op = np.asarray(df[1])
    err = (op - (ip.dot(cf[1])+cf[0]))**2
    return np.sum(err)/len(ip)


# In[34]:


w = learn(train,valid, 10**(-6),10**(-4))


# In[24]:


print ("Test MSE:", mse(w,test))


# In[49]:


plt.subplot(2,1,1)
plt.plot(train[0],train[1], 'ro')
xaxis = np.arange(np.amin(train[0]),np.amax(train[0]),0.005)
plt.plot(xaxis,w[0] + np.dot(xaxis,w[1]))
plt.xlabel("x value")
plt.ylabel("y value")
plt.title("Training Set")
plt.subplot(2,1,2)
plt.plot(valid[0],valid[1], 'ro')
xaxisv = np.arange(np.amin(valid[0]),np.amax(valid[0]),0.005)
plt.plot(xaxisv,w[0] + np.dot(xaxisv,w[0]))
plt.xlabel("x value")
plt.ylabel("y value")
plt.title("Validation Set")


# In[52]:


def learn_visual(train,valid,a,threshold):
    ip = np.asarray(train[0])
    op = np.asarray(train[1])
    cfPrev = np.array([1,1],dtype=float)
    cf = np.array([0,0],dtype=float)
    index = np.arange(0,len(ip))
    epoch = 0
    
    plt.plot(train[0],train[1], 'yo')
    xaxis = np.arange(np.amin(train[0]),np.amax(train[0]),0.005)
        
    while (np.linalg.norm(cf - cfPrev)) > threshold:
        epoch+=1
        random.shuffle(index)
        cfPrev[0] = cf[0]
        cfPrev[1] = cf[1]
        for i in range(len(ip)):
            j  = index[i]
            cf[0] = cf[0] - a*(cf[0]+cf[1]*ip[j] - op[j])
            cf[1] = cf[1] - a*(cf[0]+cf[1]*ip[j] - op[j])*ip[j]        
        
        if epoch < 5 or epoch == 150:
            plt.plot(xaxis,cf[0] + np.dot(xaxis,cf[1]))  
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.title("Fitting Evolution")
    plt.legend(['Training set', 'epoch = 1', 'epoch = 2', 'epoch = 3', 'epoch = 4', 'epoch = 150'])
    plt.show()
    print ("Fianl Validation MSE:", mse(cf,valid))
    print ("# Epoch:" , epoch)
    return cf

learn_visual(train,valid,10**(-3),10**(-4))

