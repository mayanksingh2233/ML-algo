#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets


# In[3]:


df=datasets.load_digits()


# In[4]:


df.data[0]


# In[7]:


plt.gray()
for i in range(0,5):
    plt.matshow(df.images[i])


# In[8]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(df.data,df.target,test_size=0.2)


# In[12]:


len(x_test)


# In[13]:


len(x_train)


# In[14]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[15]:


model.fit(x_train,y_train)


# In[16]:


model.score(x_test,y_test)


# In[17]:


plt.matshow(df.images[67])


# In[18]:


df.target[67]


# In[19]:


model.predict([df.data[67]])


# In[20]:


from sklearn.metrics import confusion_matrix


# In[24]:


y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)


# In[23]:


import seaborn as sns


# In[25]:


plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('actual')


# In[ ]:




