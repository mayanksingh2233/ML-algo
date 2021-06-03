#!/usr/bin/env python
# coding: utf-8

# # logistic regression

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[7]:


df=pd.read_csv('E:\\python\\datasets\\logistic.csv')
df.head()


# In[9]:


y=df[['Purchased']]


# In[10]:


x=df[['EstimatedSalary']]


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


lr=LogisticRegression()


# In[17]:


lr.fit(x_train,y_train)


# In[18]:


y_pred=lr.predict(x_test)


# In[19]:


y_pred


# In[20]:


y_test


# In[21]:


from sklearn.metrics import confusion_matrix


# In[22]:


confusion_matrix(y_test,y_pred)


# In[23]:


(82+0)/(38+0+82+0)


# In[25]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color='r')
plt.show()


# In[ ]:




