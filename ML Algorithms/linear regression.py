#!/usr/bin/env python
# coding: utf-8

# # linear regression

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[115]:


df=pd.read_csv('E:\\python\\datasets\\cars.csv',delimiter=';',skiprows=[1])
df.head()


# In[118]:


x=df[['Displacement']]
x


# In[120]:


y=df[['Acceleration']]
y


# In[121]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[122]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[123]:


lr=LinearRegression()


# In[124]:


lr.fit(x_train,y_train)


# In[125]:


y_pred=lr.predict(x_test)


# In[126]:


y_pred


# In[127]:


y_test


# In[110]:


from sklearn.metrics import mean_squared_error


# In[128]:


mean_squared_error(y_pred,y_test)


# In[130]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color='r')
plt.show()


# In[ ]:




