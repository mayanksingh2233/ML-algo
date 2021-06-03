#!/usr/bin/env python
# coding: utf-8

# # knn
# 

# In[1]:


import pandas as pd
df=pd.read_csv('iris.csv')


# In[2]:


df.head()


# In[5]:


x=df.iloc[:,:4]
y=df.iloc[:,4:5]


# In[6]:


from sklearn.neighbors import KNeighborsClassifier


# In[7]:


clf=KNeighborsClassifier()
clf.fit(x,y)


# In[8]:


clf.predict([[2.3,3.4,4.3,1.3]])


# In[9]:


clf.score(x,y)


# In[ ]:




