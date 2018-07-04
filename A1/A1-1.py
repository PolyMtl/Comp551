
# coding: utf-8

# In[105]:


import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt('Dataset_1_train.csv', delimiter=',', usecols=(0,1))
valid = np.genfromtxt('Dataset_1_valid.csv', delimiter=',', usecols=(0,1))
test = np.genfromtxt('Dataset_1_test.csv', delimiter=',', usecols=(0,1))


# In[106]:


def phi(ip, deg):
#     return list(map(lambda x,i: list(map(lambda j: x[i]**j, range(0,deg+1))), ip, range(0,ip.shape[0])))
    xi = np.zeros(len(ip)*(deg+1)).reshape(len(ip),(deg+1))
    for i in range(0,len(ip)):
        for j in range(0,deg+1):
            xi[i][j] = ip[i]**j
    return xi

def training(df, deg):
    ip = df[:,0]
    op = df[:,1]
    x = phi(ip,deg)
    cf = np.linalg.inv((x.transpose().dot(x))).dot(x.transpose()).dot(op)
    
    t = np.arange(np.amin(ip),np.amax(ip),0.001)
    plt.plot(ip,op,"ro")
    plt.plot(t, phi(t,deg).dot(cf))
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.title("Training set")
    plt.show()
    print("Training MSE:", mse(cf,x,op))
    return cf

def validation(df, cf):
    ip = df[:,0]
    op = df[:,1]
    x = phi(ip,len(cf)-1)
    t = np.arange(np.amin(ip),np.amax(ip),0.001)
    plt.plot(ip,op,"ro")
    plt.plot(t, phi(t,len(cf)-1).dot(cf))
    plt.xlabel("x value")
    plt.ylabel("y value")
#     plt.title("Validation set")
#     plt.show()
#     print("Validation MSE:", mse(cf,x,op))
    return mse(cf,x,op)

def mse(cf,x, y):
    err = (y - x.dot(cf)).transpose().dot(y - x.dot(cf))
    m = err/len(y)
    return m


# In[111]:


coeff = training(train,20)
m = validation(valid,coeff)
plt.title("Validation set")
plt.show()
print("Validation MSE:", m)


# In[112]:


def l2reg(df, deg, ld):
    ip = df[:,0]
    op = df[:,1]
    x = phi(ip,deg)
    cf = np.linalg.inv((x.transpose().dot(x)) + np.identity(deg+1).dot(ld)).dot(x.transpose()).dot(op)
#     plt.plot(ld,mse(cf,x,op),"ro", label="training")
#     print("L2 Training MSE:", mse(cf,x,op))
    return cf


# In[113]:


def testLd(df,deg,l):
    bestld = np.nan
    bestMSE = 10
    for ld in np.arange(0.0001,l,0.01):
        cf = l2reg(df,deg,ld)
        trainM, = plt.plot(ld,mse(cf,phi(train[:,0],deg),train[:,1]),"ro", label="training")
        vMSE = mse(cf,phi(valid[:,0],deg),valid[:,1])
        if bestMSE > vMSE:
            bestMSE = vMSE
            bestld = ld
        validM, = plt.plot(ld,vMSE,"bo", label = "validation")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.title("L2 Regularization")
    plt.legend([trainM,validM],["Training MSE","Validation MSE"])
    plt.show()
    
    return bestld

bLd = testLd(train,20,1)
print("Best lambda:", bLd)


# In[116]:


# test performance
cf = l2reg(test, 20, bLd)
m = validation(test,cf)
plt.title("Test performance")
plt.show()
print("Test MSE:", m)

