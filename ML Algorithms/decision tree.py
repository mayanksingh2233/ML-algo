#!/usr/bin/env python
# coding: utf-8

#  # decision tree

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


df=pd.read_csv('E:\\python\\datasets\\iris.csv')
df.head()


# In[11]:


y=df[['species']]


# In[10]:


x=df[['sepal_length']]


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dtc=DecisionTreeClassifier()


# In[16]:


dtc.fit(x_train,y_train)


# In[19]:


y_pred=dtc.predict(x_test)


# In[20]:


y_pred


# In[16]:


y_test.head()


# In[50]:


from sklearn.metrics import accuracy_score


# In[18]:


accuracy_score(y_test,y_pred)


# In[28]:


from sklearn import tree


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


from sklearn.tree import plot_tree


# In[49]:


plt.figure(figsize=(25,10))
a=plot_tree(dtc,filled=True,rounded=True,fontsize=14)

