#!/usr/bin/env python
# coding: utf-8

# # KNN prediction
# 

# In[6]:


import numpy as np
from sklearn import neighbors,datasets


# In[7]:


iris=datasets.load_iris()


# In[9]:


x=iris.data[:,:2]
y=iris.target
h=.02


# In[10]:


clf=neighbors.KNeighborsClassifier(n_neighbors=6,weights='distance')


# In[11]:


clf.fit(x,y)


# In[13]:


s1=input('enter sepal_lenght(cm):')
s2=input('enter sepal_width(cm):')
dataclass=clf.predict([[s1,s2]])
print('prediction:')
if dataclass==0:
    print('iris setosa')
elif dataclass==1:
    print('iris versicolor')
else:
    print('iris virginica')
    


# In[ ]:




