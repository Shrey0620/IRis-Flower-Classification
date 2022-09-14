#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("iris.csv")


# In[3]:


df.head()


# In[4]:


#to display statistics about data
df.describe()


# In[5]:


#to display basic info about datatyes
df.info()


# In[6]:


#to display no of samples of each class
df['Species'].value_counts()


# **Preprocessing the dataset**
# 
# 

# In[7]:


#check for null values
df.isnull().sum()


# # **Exploratory Data Analysis**

# In[8]:


df['SepalLengthCm'].hist()


# In[9]:


df['SepalWidthCm'].hist()


# In[10]:


df['PetalLengthCm'].hist()


# In[11]:


df['PetalWidthCm'].hist()


# In[12]:


#scatterplot
colors = ['red', 'green', 'blue']
species = ['Iris-virginica','Iris-vesicolor','Iris-setosa']


# In[13]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c=colors[i], label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()


# In[14]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c=colors[i], label=species[i])
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()


# In[15]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'], c=colors[i], label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.legend()


# In[16]:


for i in range(3):
    x=df[df['Species']==species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalWidthCm'], c=colors[i], label=species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Width")
    plt.legend()


# In[17]:


sns.pairplot(df,hue = 'Species', palette='deep')
plt.show()


# In[18]:


plt.figure(figsize = (10,8))
sns.heatmap(df.corr(), annot = True, cmap = 'YlGnBu')
plt.show()


# In[19]:


x=df.drop(['Species'], axis = 1)


# In[20]:


y=df['Species']


# In[21]:


x


# In[22]:


y


# In[23]:


df.corr()


# In[24]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True, ax=ax, cmap= 'coolwarm')


# # **Label Encoder**

# In[25]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[26]:


df['Species']= le.fit_transform(df['Species'])
df.head()


# # Model training logistic regression

# In[27]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y= lb.fit_transform(y)


# In[28]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()


# In[29]:


sc


# In[30]:


x_std = sc.fit_transform(x)


# In[31]:


from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=0.3, random_state=0)


# In[32]:


from sklearn import linear_model


# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


lr = LogisticRegression(random_state=0, multi_class="ovr")


# In[35]:


lr.fit(x_std,y)


# In[36]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[37]:


y_pred = lr.predict(x_std)


# In[38]:


from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred)*100


# In[ ]:




