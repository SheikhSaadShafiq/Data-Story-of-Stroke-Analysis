#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot


# In[4]:


path = "/Users/saadshafiq/Desktop/IDS/ids_project/stroke_dataset.csv"
df = pd.read_csv(path)
df.head(15)


# In[5]:


# inspiration code: https://www.kaggle.com/dwin183287/covid-19-world-vaccination
fig=plt.figure(figsize=(5,2),facecolor='white')

ax0=fig.add_subplot(1,1,1)
ax0.text(1.1,1,"Key figures",color='black',fontsize=28, fontweight='bold', fontfamily='monospace',ha='center')

ax0.text(0,0.4,"5000",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(0,0.001,"Number of patients \nin the dataset",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(0.75,0.4,"12",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(0.75,0.001,"Number of features \nin the dataset",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(1.5,0.4,"43.6k",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(1.5,0.001,"Downloads \nof this Dataset",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.text(2.25,0.4,"Rural/Urban",color='gold',fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(2.25,0.001,"Diversion \nof data",color='dimgrey',fontsize=17, fontweight='light', fontfamily='monospace',ha='center')

ax0.set_yticklabels('')
ax0.tick_params(axis='y',length=0)
ax0.tick_params(axis='x',length=0)
ax0.set_xticklabels('')

for direction in ['top','right','left','bottom']:
    ax0.spines[direction].set_visible(False)


# # Data Wrangling 
# ### Identify and handle missing values
# 
# 1. Remove Empty Value
# 2. Remove NaN

# In[6]:


#replacing ? values with NAN

df.replace("?",np.nan,inplace = True)
len(df)


# In[7]:


#dropping NAN values because i have a 5,000+ dataset 
df.dropna(subset=['bmi'], axis=0, inplace=True)

#dropping Unkown values in smoking_status
df.drop(df[df['smoking_status'] == 'Unknown'].index, inplace = True)
len(df)
df.head(100)


# In[8]:


#Converting datatypes
df.dtypes 


# In[9]:


df[['gender','ever_married','work_type','Residence_type','smoking_status']] = df[['gender','ever_married','work_type','Residence_type','smoking_status']].astype('str')
df[['age']] = df[['age']].astype('int64')
df[['avg_glucose_level','bmi']] = df[['avg_glucose_level','bmi']].astype('float128')
df.dtypes


# In[10]:


df.to_csv("/Users/saadshafiq/Desktop/IDS/ids_project/test.csv")
df


# #### Data Wrangling DONE!
# 
# It's done because the i removed the columns directly from it. Because i had the 5000+ person dataset. In this case i am just testing and predicting different scearnios. 
# I removed it and now i've 
# ###### 3426 * 12 in the dataset

# In[11]:


#Univariate analysis of continuous variables

fig=plt.figure(figsize=(20,8),facecolor='white')
gs=fig.add_gridspec(1,2)
ax=[None for i in range(2)]
ax[0]=fig.add_subplot(gs[0,0])
ax[1]=fig.add_subplot(gs[0,1])

ax[0].text(-24,0.025,'Distribution of the age variable',fontsize=23,fontweight='bold', fontfamily='monospace')
ax[0].text(-24,0.0238,'Most of the people in the dataset are between 40 to 60 years old',fontsize=17,fontweight='light', fontfamily='monospace')

ax[1].text(6,375,'Distribution of the bmi variable',fontsize=23,fontweight='bold', fontfamily='monospace')
ax[1].text(6,355,'Most of the people in the dataset are between 25 to 35 of bmi',fontsize=17,fontweight='light', fontfamily='monospace')

sns.kdeplot(x=df['age'],ax=ax[0],shade=True, color='gold', alpha=0.6,zorder=3,linewidth=5,edgecolor='black')
sns.histplot(x=df['bmi'],ax=ax[1], color='gold', alpha=1,zorder=2,linewidth=1,edgecolor='black',shrink=0.5)

for i in range(2):
    ax[i].set_ylabel('')
    ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)
        
ax[1].set_xlim(10,70)
        
plt.tight_layout()


# ### My Questions:
# 
# 
# 1. What is the relation between the age and Hypertension of a normal person and a heart disease patient? --> (age,hypertension, heart disease)
# 2. Does Marriage plays a part in getting heart disease? -->  (Marriage, Heart Disease)
# 3. Does work type affects mental peace?/Which work type doesn't affect your mental peace? --> (work types, mental peace) 
# 4. What are the average glucose level of a normal person and a heart disease patient? --> (average glucose level, normal person / heart disease patient)
# 5. •	What body mass index defines a patient life or a healthy life? --> (body mass, heart_disease = 1)

# ## Question 1
# ##### what is the relation between the age and Hypertension of a normal person and a heart disease patient?

# In[11]:


#filtering heart patient seprate and normal person seprate 

heart = df[["id", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
            "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"]] [df.heart_disease == 1]
non_heart =  df[["id", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
            "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"]] [df.heart_disease == 0]


# In[13]:


fig=plt.figure(figsize=(20,8),facecolor='white')

gs=fig.add_gridspec(1,2)

ax=[_ for _ in range(1)]
ax[0]=fig.add_subplot(gs[0,0])


ax[0].text(-1,15,"Relationship between age and Hypertension",fontsize=21,fontweight='bold', fontfamily='monospace')

sns.lineplot(data=non_heart,x='age',y='hypertension',ax=ax[0],color='goldenrod')


for i in range(1):
    
    ax[i].set_ylabel('Hypertension').set_rotation(0)
    ax[i].set_yticklabels('')
    ax[i].tick_params(axis='y',length=0)
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)
        
plt.tight_layout()
plt.show()


# In[14]:


fig=plt.figure(figsize=(20,8),facecolor='white')

gs=fig.add_gridspec(1,2)

ax=[_ for _ in range(1)]
ax[0]=fig.add_subplot(gs[0,0])


ax[0].text(1,15,"Relationship between age and hypertension of a heart dieases Patient",fontsize=21,fontweight='bold', fontfamily='monospace')

sns.lineplot(data=heart,x='age',y='hypertension',ax=ax[0],color='goldenrod')

for i in range(1):
    
    ax[i].set_ylabel('Hypertension').set_rotation(0)
    ax[i].set_yticklabels('')
    ax[i].tick_params(axis='y',length=0)
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)
        
plt.tight_layout()
plt.show()


# So as the age is going up the Hypertension is going up in a normal person While in heart patient, hypertension is there but that is not increasing much as compared to younger people on non-heart patient people.

# ## Question 2
# ##### Does Marriage plays a part in getting heart disease?

# In[16]:



nm = df[['ever_married','heart_disease']] [df.heart_disease==0] [df.ever_married == "No"]
m = df[['ever_married','heart_disease']] [df.heart_disease==1] [df.ever_married == "Yes"]
a = len(nm)
b = len(m)
total = a+b
y = np.array([a, b])
myexplode = [0.25, 0]
mylabels = ["Doesn't play a part", "Yes, It Does"]
mycolors = ["palegreen", "green"]

plt.pie(y,explode = myexplode, labels = mylabels, colors = mycolors)
plt.show() 
a= (a*100)/total
b = (b*100)/total

print(a, "%\t\tData shows Marriage doesn't play a part in getting heart Disease\n")
print(b, "%\t\tData shows Marriage play a part in getting heart Disease\n")


# In[12]:


fig=plt.figure(figsize=(20,8),facecolor='white')

ax=[None for i in range(2)]
gs=fig.add_gridspec(2,1)
gs.update(wspace=0, hspace=0.8)

ax[0]=fig.add_subplot(gs[0,0])
ax[1]=fig.add_subplot(gs[1,0])

ax[0].text(-20,0.04,'Relationship between age and stroke',fontsize=21,fontweight='bold', fontfamily='monospace')
ax[0].text(-20,0.035,'The older a person is, the more likely he/she has a stroke',fontsize=16,fontweight='light', fontfamily='monospace')
ax[1].text(-80,0.023,'Relationship between average glucose level and stroke',fontsize=21,fontweight='bold', fontfamily='monospace')
ax[1].text(-80,0.0207,'From this graph, there is no clear relationship between avg_glucose_level and stroke',fontsize=16,fontweight='light', fontfamily='monospace')

sns.kdeplot(data=df[df.stroke==1],x='age',ax=ax[0],shade=True,color='lightcoral',alpha=1)
sns.kdeplot(data=df[df.stroke==0],x='age',ax=ax[0],shade=True,color='palegreen',alpha=0.5)
sns.kdeplot(data=df[df.heart_disease==1],x='avg_glucose_level',ax=ax[1],shade=True,color='lightcoral',alpha=1)
sns.kdeplot(data=df[df.heart_disease==0],x='avg_glucose_level',ax=ax[1],shade=True,color='palegreen',alpha=0.5)

for i in range(2):
    ax[i].set_yticklabels('')
    ax[i].set_ylabel('')
    ax[i].tick_params(axis='y',length=0)
    
    for direction in ['top','right','left']:
        ax[i].spines[direction].set_visible(False)


# ## Question 3
# ###### Does work type affects mental peace?/Which work type doesn't affect your mental peace?

# In[13]:


import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df['work_type'])
plt.pyplot.xlabel('Work Types',fontweight='bold', fontfamily='monospace')
plt.pyplot.ylabel('Total Count',fontweight='bold', fontfamily='monospace')
plt.pyplot.title('Which work type has a higher rank',fontsize=15,fontweight='bold', fontfamily='monospace')
plt.pyplot.ylim(0,2500)


#  So as we can see that Private work type has a higher rank as compared to others.

# ### Question 4 
# ###### What are the average glucose level of a normal person and a heart disease patient? 

# In[14]:


h1 = df[["avg_glucose_level"]] [df.heart_disease == 1]
h0 = df[["avg_glucose_level"]] [df.heart_disease == 0]


plt.pyplot.hist(h1['avg_glucose_level'].mean(), histtype='step', color = 'r')
plt.pyplot.hist(h0['avg_glucose_level'].mean(), histtype='step',color = 'g')
plt.pyplot.xlabel('Glucose Level',fontsize=18,fontweight='light', fontfamily='monospace')
plt.pyplot.title('\n\nAverage Gluscose Level',fontsize=15,fontweight='bold', fontfamily='monospace')
#\nGreen color represents Normal Patient 107\nRed Color represents Heart Disease patients 135


# So As we can see that Heart Disease patient has a higher gluscose level on average.

# ## Question 5 
# 
# ###### What body mass index defines a patient life or non-patient life?

# In[15]:


fig=plt.figure(figsize=(20,8),facecolor='white')

ax=[None for i in range(2)]
gs=fig.add_gridspec(2,1)
gs.update(wspace=0, hspace=0.8)

ax[0]=fig.add_subplot(gs[0,0])
ax[1]=fig.add_subplot(gs[1,0])

ax[0].text(-20,0.095,'Relationship between Heart Diesease and BMI',fontsize=25,fontweight='bold', fontfamily='monospace')
ax[0].text(-20,0.085,'It shows that both type of people has almost same body mass',fontsize=16,fontweight='light', fontfamily='monospace')

sns.kdeplot(data=df[df.heart_disease==1],x='bmi',ax=ax[0],shade=True,color='lightcoral',alpha=1)
sns.kdeplot(data=df[df.heart_disease==0],x='bmi',ax=ax[0],shade=True,color='palegreen',alpha=0.5)


#for i in range(1):
ax[i].set_yticklabels('')
ax[i].set_ylabel('')
ax[i].tick_params(axis='y',length=0)
    
# for direction in ['top','right','left']:
ax[i].spines[direction].set_visible(False)


#  So it shows that both type of people has almost same body type.

# ## The End 

# # Modeling 

# In[14]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
  
df['gender']= le.fit_transform(df['gender'])
df['ever_married']= le.fit_transform(df['ever_married'])
df['work_type']= le.fit_transform(df['work_type'])
df['Residence_type']= le.fit_transform(df['Residence_type'])
df['smoking_status']= le.fit_transform(df['smoking_status'])
df.head()


# In[21]:


corr = df.corr().round(2)
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot = True, cmap = 'RdYlGn');


# In[18]:


x = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) 
print("x_train1", x_train) 


# In[ ]:





# In[ ]:





# In[21]:


# Feature Scaling  
print("feature scaling")
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test) 
print("x_train", x_train) 


# In[ ]:





# In[22]:


# Fitting Naive Bayes to the Training set  
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()  
classifier.fit(x_train, y_train) 


# In[23]:


# Predicting the Test set results  
y_pred = classifier.predict(x_test)  
print(y_pred)


# In[24]:


## Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy
print("\nAccuracy: ", np.around(accuracy_score(y_test, y_pred), 2)*100, "%")


# In[ ]:





# In[ ]:




