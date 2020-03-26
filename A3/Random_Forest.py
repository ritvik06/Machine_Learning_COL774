#!/usr/bin/env python
# coding: utf-8

# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pprint
import matplotlib.pyplot as plt


# In[4]:


#####################################  LOADING TRAIN X DATA   ####################################
f_x = open('./Data/train_x.txt','r')

X_train = np.zeros((64713,482))

line1 = f_x.readline()
counter = 0


while True:
    line1 = f_x.readline()
    
    if not line1:
        break
        
    counter+=1
    
    arr = line1.strip().split(' ')
    
    for i in range(len(arr)):
        index = int(arr[i].split(':')[0])
        value = int(float(arr[i].split(':')[1]))
        
        X_train[counter-1][index] = value
        
#####################################  LOADING TRAIN Y DATA   ####################################

f_y = open('./Data/train_y.txt','r')

Y_train = []

while True:
    line2 = f_y.readline()
    
    if not line2:
        break
        
    Y_train.append(int(line2))

Y_train = np.asarray(Y_train)


#####################################  LOADING VAL X DATA   ####################################

val_x = open('./Data/valid_x.txt','r')

X_val = np.zeros((21572,482))

line1 = val_x.readline()
counter = 0


while True:
    line1 = val_x.readline()
    
    if not line1:
        break
        
    counter+=1
    
    arr = line1.strip().split(' ')
    
    for i in range(len(arr)):
        index = int(arr[i].split(':')[0])
        value = int(float(arr[i].split(':')[1]))
        
        X_val[counter-1][index] = value
        
#####################################  LOADING VAL Y DATA   ####################################

val_y = open('./Data/valid_y.txt','r')

Y_val = []

while True:
    line2 = val_y.readline()
    
    if not line2:
        break
        
    Y_val.append(int(line2))

Y_val = np.asarray(Y_val)


#####################################  LOADING TEST X DATA   ####################################

test_x = open('./Data/test_x.txt','r')

X_test = np.zeros((21571,482))

line = test_x.readline()
counter = 0


while True:
    line1 = test_x.readline()
    
    if not line1:
        break
        
    counter+=1
    
    arr = line1.strip().split(' ')
    
    for i in range(len(arr)):
        index = int(arr[i].split(':')[0])
        value = int(float(arr[i].split(':')[1]))
        
        X_test[counter-1][index] = value
        
#####################################  LOADING TEST Y DATA   ####################################

test_y = open('./Data/test_y.txt','r')

Y_test = []

while True:
    line2 = test_y.readline()
    
    if not line2:
        break
        
    Y_test.append(int(line2))

Y_test = np.asarray(Y_test)


# In[16]:


rfc = RandomForestClassifier(oob_score=True)
parameters = {'n_estimators':[50, 150, 250, 350, 450], 'max_features':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'min_samples_split':[2, 4, 6, 8, 10]}


# In[17]:


def scorer(estimator,X,Y):
    return estimator.oob_score_


# In[20]:


search = GridSearchCV(estimator=rfc, param_grid=parameters, scoring=scorer,n_jobs=-1,cv=2)
search.fit(X_train, Y_train)


# In[23]:


res = search.predict(X_train)
correct = 0

for i in range(len(res)):
    if res[i]==Y_train[i]:
        correct+=1
        
print('Training Accuracy is ', (correct/len(res))*100,'%')


# In[22]:


res = search.predict(X_val)
correct = 0

for i in range(len(res)):
    if res[i]==Y_val[i]:
        correct+=1
        
print('Val Accuracy is ', (correct/len(res))*100,'%')


# In[24]:


res = search.predict(X_test)
correct = 0

for i in range(len(res)):
    if res[i]==Y_test[i]:
        correct+=1
        
print('Test Accuracy is ', (correct/len(res))*100,'%')


# In[ ]:


print(clf.best_params_)


# In[ ]:


estimator = clf.best_estimator_


# In[ ]:


print(estimator.oob_score_) #oob accuracy


# In[8]:


train_acc_estimators = []
val_acc_estimators = []
test_acc_estimators = []

n_estimator = [50, 150, 250, 350, 450]
for i in range(len(n_estimator)):
    estimator = RandomForestClassifier(n_estimators=n_estimator[i], oob_score=True, min_samples_split=10, max_features=0.1, n_jobs=-1)
    estimator.fit(X_train, Y_train)
    train_acc_estimators.append(estimator.score(X_train, Y_train))
    val_acc_estimators.append(estimator.score(X_val, Y_val))
    test_acc_estimators.append(estimator.score(X_test, Y_test))

    print(n_estimator[i], estimator.oob_score_, test_acc_estimators[i], val_acc_estimators[i], train_acc_estimators[i])


# In[11]:


train_acc_max = []
val_acc_max = []
test_acc_max = []

max_feature = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

for i in range(len(max)):
    estimator = RandomForestClassifier(n_estimators=350, oob_score=True, min_samples_split=10, max_features=max_feature[i], n_jobs=-1)
    estimator.fit(X_train, Y_train)
    train_acc_max.append(estimator.score(X_train, Y_train))
    val_acc_max.append(estimator.score(X_val, Y_val))
    test_acc_max.append(estimator.score(X_test, Y_test))

    print(max_feature[i], estimator.oob_score_, test_acc_max[i], val_acc_max[i], train_acc_max[i])


# In[26]:


train_acc_split = []
val_acc_split = []
test_acc_split = []

split = [2, 4, 6, 8, 10]

for i in range(len(split)):
    estimator = RandomForestClassifier(n_estimators=350, oob_score=True, min_samples_split=split[i], max_features=0.1, n_jobs=-1)
    estimator.fit(X_train, Y_train)
    train_acc_split.append(estimator.score(X_train, Y_train))
    val_acc_split.append(estimator.score(X_val, Y_val))
    test_acc_split.append(estimator.score(X_test, Y_test))

    print(split[i], estimator.oob_score_, test_acc_split[i], val_acc_split[i], train_acc_split[i])


# In[40]:


plt.figure()
plt.plot(n_estimator, train_acc_estimators[:5], label='train set',color='b')
plt.plot(n_estimator, val_acc_estimators[:5], label='valid set',color='r')
plt.plot(n_estimator,test_acc_estimators[:5], label='test set',color='g')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.figure()
plt.plot(n_estimator, val_acc_estimators[:5], label='valid set',color='r',marker='o')
plt.plot(n_estimator,test_acc_estimators[:5], label='test set',color='g',marker='<')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.ylim(0.78,0.82)
plt.legend()
plt.show()


# In[35]:


plt.figure()
plt.plot(max_feature, train_acc_max, label='train set',color='b')
plt.plot(max_feature, val_acc_max, label='valid set',color='r')
plt.plot(max_feature,test_acc_max, label='test set',color='g')
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(max_feature, val_acc_max, label='valid set',color='r',marker='o')
plt.plot(max_feature,test_acc_max, label='test set',color='g',marker='<')
plt.xlabel('max_features')
plt.ylabel('Accuracy')
# plt.ylim(0.78,0,82)
plt.legend()
plt.show()


# In[38]:


plt.figure()
plt.plot(split, train_acc_split, label='train set',color='b')
plt.plot(split, val_acc_split, label='valid set',color='r')
plt.plot(split,test_acc_split, label='test set',color='g')
plt.xlabel('min samples split')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(split, val_acc_split, label='valid set',color='r',marker='o')
plt.plot(split,test_acc_split, label='test set',color='g',marker='<')
plt.xlabel('min samples split')
plt.ylabel('Accuracy')
plt.ylim(0.78,0.82)
plt.legend()
plt.show()


# In[ ]:




