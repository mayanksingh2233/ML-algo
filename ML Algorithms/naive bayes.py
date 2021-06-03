#!/usr/bin/env python
# coding: utf-8

# # Naive bayes
# 

# In[17]:


import pandas as pd
data=pd.read_csv('iris.csv')
data.head()


# In[18]:


y=data[['species']]


# In[20]:


x=data[['sepal_length']]


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[23]:


from sklearn.naive_bayes import GaussianNB


# In[24]:


gnb=GaussianNB()


# In[25]:


gnb.fit(x_train,y_train)


# In[26]:


y_pred=gnb.predict(x_test)


# In[31]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[32]:


from sklearn.metrics import accuracy_score


# In[38]:


print('classification report:\n',classification_report(y_test,y_pred))
print('confusion matrix:\n',confusion_matrix(y_test,y_pred))
print('model accuracy:',accuracy_score(y_test,y_pred))


# In[ ]:




