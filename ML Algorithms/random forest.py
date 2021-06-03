#!/usr/bin/env python
# coding: utf-8

# # Random forest

# In[3]:


import pandas as pd
from sklearn import datasets


# In[10]:


df=datasets.load_digits()
dir(df)


# In[12]:


x=df.data
x


# In[13]:


y=df.target


# In[14]:


y


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[17]:


rfc=RandomForestClassifier(n_estimators=100)


# In[18]:


rfc.fit(x_train,y_train)


# In[19]:


rfc.score(x_test,y_test)


# In[20]:


rfc.predict(x_test)


# In[21]:


y_test


# In[ ]:




