#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('iris.csv')
df.head()


# In[8]:


x=df.iloc[:,:4].values


# In[7]:


y=df.iloc[:,4:5].values


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[12]:


sr=SVC()


# In[13]:


sr.fit(x_train,y_train)


# In[14]:


sr.score(x_test,y_test)


# In[15]:


sr.predict([[2.3,3.4,4.3,1.3]])


# In[ ]:





# In[ ]:




