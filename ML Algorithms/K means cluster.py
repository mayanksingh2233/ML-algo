#!/usr/bin/env python
# coding: utf-8

# # K means clustering 

# In[2]:


import pandas as pd
df=pd.read_csv('iris.csv')
df.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder
le_species=LabelEncoder()
df['species_n']=le_species.fit_transform(df['species'])
df.head()


# In[8]:


import matplotlib.pyplot as plt
plt.scatter(df['petal_length'],df['petal_width'])
plt.show()


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


km=KMeans(n_clusters=2)
km


# In[12]:


y_pred=km.fit_predict(df[['petal_length','petal_width']])


# In[13]:


y_pred


# In[14]:


df['cluster']=y_pred


# In[15]:


df.head()


# In[24]:


km.cluster_centers_


# In[26]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
plt.scatter(df1.petal_length,df1.petal_width,color='green')
plt.scatter(df2.petal_length,df2.petal_width,color='red')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='blue',marker='+')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.legend()


# In[29]:


k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['petal_length','petal_width']])
    sse.append(km.inertia_)


# In[30]:


sse


# In[32]:


plt.xlabel('K')
plt.ylabel('sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:




