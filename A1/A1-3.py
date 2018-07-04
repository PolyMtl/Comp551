
# coding: utf-8

# In[246]:


import numpy as np
import matplotlib.pyplot as plt
import random

data = np.genfromtxt('communities.data', delimiter=',')
# data.shape = (1994,128)


# In[247]:


# throw out first five unpredictive data
data_trim = data[:,5:]
#data_trim.shape = (1994,123)

# find nan
nIndex = np.argwhere(np.isnan(data_trim))

# complete data
for i in nIndex:
    ad = data_trim[np.argwhere(~np.isnan(data_trim[:,i[1]])),i[1]]
    m  = np.mean(ad)
    data_trim[i[0],i[1]] = m


# In[258]:


def learn(df):
    ip = np.asmatrix(df[:,0:df.shape[1]-2])
    xi = np.append(ip,np.ones(ip.shape[0]).reshape(ip.shape[0],1),axis=1)
    op = np.asarray(df[:,df.shape[1]-1])
    coeff = np.linalg.inv((xi.transpose()).dot(xi)).dot(xi.transpose()).dot(op)
    return coeff

def mse(cf, df):
    ip = np.asmatrix(df[:,0:df.shape[1]-2])
    xi = np.append(ip,np.ones(ip.shape[0]).reshape(ip.shape[0],1),axis=1)
    op = np.asarray(df[:,df.shape[1]-1]).reshape(df.shape[0],1)
    err = (op - xi.dot(cf.transpose())).transpose().dot(op - xi.dot(cf.transpose()))
    return np.sum(err)/ip.shape[0]

# create training set and validation set
def cross(df, k):
    # of examples = 1994
    nEg = df.shape[0]
    
    index = np.arange(0,nEg)
    random.shuffle(index)
    part = [df[index[0: np.ceil(nEg/k).astype(int)]], df[index[np.ceil(nEg/k).astype(int): np.ceil(2*nEg/k).astype(int)]], df[index[np.ceil(2*nEg/k).astype(int): np.ceil(3*nEg/k).astype(int)]], df[index[np.ceil(3*nEg/k).astype(int): np.ceil(4*nEg/k).astype(int)]], df[index[np.ceil(4*nEg/k).astype(int):]]]
    
    for i in range(0,k):
#         random.shuffle(index)
        valid = part[i]
        train = np.concatenate(np.delete(part,i))
        name = 'CandC-train<'+str(i)+'>.csv'
        np.savetxt(name,train,delimiter=',')
        name = 'CandC-test<'+str(i)+'>.csv'
        np.savetxt(name,valid,delimiter=',')
        w = learn(train)

        print(mse(w,valid))


# In[263]:


allMse = cross(data_trim, 5)
print(allMse)


# In[262]:


def l2reg(df,ld):
    ip = np.asmatrix(df[:,0:df.shape[1]-2])
    xi = np.append(ip,np.ones(ip.shape[0]).reshape(ip.shape[0],1),axis=1)
    op = np.asarray(df[:,df.shape[1]-1])
    coeff = np.linalg.inv((xi.transpose()).dot(xi)+ np.identity(xi.shape[1]).dot(ld)).dot(xi.transpose()).dot(op)
    return coeff

def cross_l2(df, k,ld):
    # num of examples = 1994
    nEg = df.shape[0]
    
    index = np.arange(0,nEg)
    random.shuffle(index)
    part = [df[index[0: np.ceil(nEg/k).astype(int)]], df[index[np.ceil(nEg/k).astype(int): np.ceil(2*nEg/k).astype(int)]], df[index[np.ceil(2*nEg/k).astype(int): np.ceil(3*nEg/k).astype(int)]], df[index[np.ceil(3*nEg/k).astype(int): np.ceil(4*nEg/k).astype(int)]], df[index[np.ceil(4*nEg/k).astype(int):]]]
    
    optCf = np.zeros(df.shape[0])
    best = 0.00001
    lMSE = 100
    for l in np.arange(0.00001,ld,0.1):
        vErr = np.zeros(5)
        for i in range(0,k):
            valid = part[i]
            train = np.concatenate(np.delete(part,i))
#             valid = df[index[0: np.ceil(nEg/k).astype(int)]]
#             train = df[index[np.ceil(nEg/k).astype(int):]]
#             name = 'CandC-train<'+str(i)+'>.csv'
#             np.savetxt(name,train,delimiter=',')
#             name = 'CandC-valid<'+str(i)+'>.csv'
#             np.savetxt(name,valid,delimiter=',')
            w = l2reg(train,l)
            vErr[i] = mse(w,valid)
        plt.plot(l,np.mean(vErr),'ro')
        if lMSE > np.mean(vErr):
            lMSE = np.mean(vErr)
            best = l
            
    plt.show()
    print("Lowest Average MSE:", lMSE)
    print("Best Lambda:", best)
    return lMSE


# In[261]:


cross_l2(data_trim, 5, 10 )


# In[257]:


# feature selection
cf = np.zeros(122)
for i in range(0,5):
    train = np.genfromtxt('CandC-train<'+str(i)+'>.csv', delimiter=',')
    valid = np.genfromtxt('CandC-test<'+str(i)+'>.csv', delimiter=',')

    cf = l2reg(train,2)+ cf

cf = cf/5
# mse(w,valid)
feature = (np.where(abs(cf)> 0.01))[1]
feature


# In[259]:


cross_l2(data_trim, 5,10)

