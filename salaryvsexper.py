#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("C:/Users/user/Desktop/Salary_Data.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x=df.iloc[:,:-1]
y=df.iloc[:,1]


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=1)


# In[12]:


x_train


# In[13]:


x_test


# In[14]:


y_train


# In[15]:


y_test


# In[16]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[17]:


y_pred=regressor.predict(x_test)
y_pred


# In[19]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("yearsofexperiencevssalary(x_train)")
plt.xlabel("yearsofexperience")
plt.ylabel("salary")


# In[ ]:




