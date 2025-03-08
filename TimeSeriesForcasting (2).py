#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from sklearn.linear_model import LinearRegression 

import warnings


from statsmodels.graphics.tsaplots import month_plot


# In[11]:


df = pd.read_csv('/Users/huzaibibnishafi/Documents/Time Series Forcasting/gold_monthly_csv.csv')


# In[12]:


df.head()


# In[13]:


df.describe()


# In[14]:


df.tail()


# In[15]:


df.shape


# In[16]:


df.columns


# In[17]:


date = pd.date_range( start = '1/1/1950' , end ='8/1/2020', freq = 'M')
date


# In[18]:


df['month'] = date 
df.drop('Date', axis = 1 , inplace = True)
df = df.set_index('month')
df.head()


# In[19]:


df.tail()


# In[20]:


df.plot(figsize=(20,8))
plt.title("Gold prices monthly since 1950")
plt.xlabel("months")
plt.ylabel("price")
plt.grid()


# In[21]:


round(df.describe())


# In[22]:


_,ax = plt.subplots(figsize=(25,8))
sns.boxplot(x= df.index.year , y= df.values[:,0],ax=ax)
plt.title("Gold price (montly) from 1950 onwards")
plt.xlabel("Year")
plt.ylabel("price")
plt.xticks(rotation = 90 )
plt.grid()


# In[27]:


fig,ax =plt.subplots(figsize=(22,8))
month_plot(df,ylabel = "gold price", ax=ax)
plt.title('Gold price per month')
plt.grid( )


# In[31]:


_,ax = plt.subplots(figsize=(25,8))
sns.boxplot(x= df.index.month_name(), y = df.values[:,0],ax=ax)
plt.title('gold price monthly')
plt.show()


# In[32]:


df_yearly_sum = df.resample('A').mean()
df_yearly_sum.plot()
plt.title("Average gold price yearly")
plt.xlabel("year")
plt.ylabel("price")


# In[37]:


df_quaterly_sum = df.resample("Q").mean()
df_quaterly_sum.plot()
plt.title("Gold prices quaterly price")
plt.xlabel("Quarterly")
plt.ylabel("Price")
plt.show()


# In[39]:


df_decade_sum =df.resample("10Y").mean()
df_decade_sum.plot()
plt.xlabel('Decade')
plt.ylabel('Price')
plt.grid()


# In[45]:


df_1 = df.groupby(df.index.year).mean().rename(columns= {"Price":"Mean"})
df_1 =df_1.merge(df.groupby(df.index.year).std().rename(columns={"Price":"Std"}),left_index = True , right_index = True)
df_1['Cov_pct']= ((df_1['Std']/df_1["Mean"])*100).round(2)
df_1.head()


# In[48]:


fig,ax = plt.subplots(figsize=(15,10))
df_1["Cov_pct"].plot()
plt.title("Avg gold price yearly since 1950")
plt.xlabel('year')
plt.ylabel('Cv in %')
plt.grid()


# In[61]:


df


# In[49]:


train = df[df.index.year <= 2015]
test = df[df.index.year > 2015]    


# In[ ]:





# In[51]:


print(train.shape)
print(test.shape)


# In[54]:


train["Price"].plot(figsize=(13,5),fontsize =15)
test["Price"].plot(figsize=(13,5),fontsize = 15)
plt.grid()
plt.legend(["Training data",'Test data'])


# In[63]:


train_time = [i+1 for i in range(len(train))]
test_time =[i+len(train)+1 for i in range(len(test))]
len(train_time),len(test_time)


# In[70]:


LR_train = train.copy()
LR_test = test.copy()


# In[71]:


LR_train['time'] = train_time
LR_test['time']= test_time 


# In[75]:


lr= LinearRegression()
lr.fit(LR_train[['time']],LR_train['Price'].values)


# In[80]:


test_predictions_model1 = lr.predict(LR_test[["time"]])
LR_test["forecast"]= test_predictions_model1


# In[81]:


plt.figure(figsize=(14,6))
plt.plot(train['Price'],label= 'train')
plt.plot(test['Price'],label = 'test')
plt.plot(LR_test['forecast'],label = 'Reg on time_test data')
plt.legend(loc= 'best')
plt.grid()


# In[93]:


def mape(actual, pred):
    return round(np.mean(abs((actual - pred) / actual)) * 100, 2)


# In[94]:


mape_model1_test = mape(test["Price"].values,test_predictions_model1)
print("MAPE is %3.3f"%(mape_model1_test),"%")


# In[98]:


results = pd.DataFrame({'Test Mape (%)':[mape_model1_test]} ,index =["RegressionOnTime"])


# In[99]:


results


# In[101]:


Naive_train = train.copy()
Naive_test = test.copy()


# In[102]:


Naive_test['naive'] =np.asarray(train['Price'])[len(np.asarray(train['Price']))-1]
Naive_test['naive'].head()


# In[107]:


plt.figure(figsize=(13,5))
plt.plot(Naive_train['Price'],label='Train')
plt.plot(Naive_test["Price"],label="Test")
plt.plot(Naive_test["naive"],label = "Naive Forecast on Test data")
plt.legend(loc="best")
plt.title("Naive Forecast")
plt.grid();


# In[109]:


mape_model2 = mape(test["Price"].values, Naive_test["naive"].values)
print("For Naive Forecast on test data, MAPE is %3.3f"%(mape_model2),"%")


# In[112]:


result2 = pd.DataFrame({"Test Mape(%)" : [mape_model2]},index =["NaiveModel"])
result3 = pd.concat([results,result2])
result3


# In[114]:


final_model = ExponentialSmoothing (df, trend="additive",
                                   seasonal= 'additive').fit(
                                            smoothing_level = 0.4,
                                                    smoothing_trend=0.3,
                                                smoothing_seasonal =0.6)


# In[115]:


Mape_final_model = mape(df["Price"].values, final_model.fittedvalues)


# In[120]:


print("Mape:" ,Mape_final_model)


# In[122]:


predictions =final_model.forecast(steps=len(test))


# In[124]:


pred_df =pd.DataFrame({"Lower_CI": predictions - 1.96*np.std(final_model.resid,ddof=1),
                       'prediction' :predictions,
                           "Upper_CI":predictions+1.96*np.std(final_model.resid,ddof=1)})


# In[125]:


pred_df.head()


# In[137]:


axis = df.plot(label = "Actual", figsize=(15,15))
pred_df['prediction'].plot(ax=axis , label ='Forecast',alpha =0.5)
axis.fill_between(pred_df.index, pred_df['Lower_CI'],pred_df['Upper_CI']
                 ,color='m', alpha =.15)
axis.set_xlabel('year-month')
axis.set_ylabel('price')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[ ]:




