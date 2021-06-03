#!/usr/bin/env python
# coding: utf-8

# # gradient&ada boost

# In[54]:


from sklearn.datasets import make_moons
mm=make_moons()


# In[8]:


x,y=make_moons(n_samples=10000,noise=.5,random_state=0)


# In[11]:


from sklearn.datasets import load_iris


# In[12]:


df=load_iris()


# In[15]:


x=df.data


# In[17]:


y=df.target
y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


from sklearn.tree import DecisionTreeClassifier


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[21]:


model=DecisionTreeClassifier()


# In[30]:


model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[25]:


y_pred=model.predict(x_test)


# In[24]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score(y_test,y_pred)


# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


rm=RandomForestClassifier(n_estimators=100,max_features='auto')


# In[41]:


rm.fit(x_train,y_train)


# In[42]:


rm.score(x_test,y_test)


# In[45]:


r_pred=rm.predict(x_test)


# In[47]:


accuracy_score(y_test,r_pred)


# In[34]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[35]:


gb=GradientBoostingClassifier(n_estimators=100)


# In[36]:


gb.fit(x_train,y_train)


# In[48]:


y_pred=gb.predict(x_test)


# In[49]:


accuracy_score(y_test,y_pred)


# In[50]:


clf=AdaBoostClassifier(n_estimators=100)


# In[51]:


clf.fit(x_train,y_train)


# In[52]:


y_pred=clf.predict(x_test)


# In[53]:


accuracy_score(y_test,y_pred)


# In[ ]:




