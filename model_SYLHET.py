#Importing_necessary_libraries_for the project
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pathlib
import os
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from __future__ import absolute_import, division, print_function, unicode_literals
from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set()

#Setting google colab environment and connecting google drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

data_dir = "/content/gdrive/My Drive/Thesis/Sylhet_Temp/SYLHET.csv"


data = data_dir

def overViewOfTheData(data,frows=5,lrows=5):
  print("Shape: ",data.shape,"\n\n")
  
  print("Columns: ",data.columns,"\n\n")

  print("Info : ")
  print(data.info())

overViewOfTheData(data)


plt.figure(figsize=(8,8))
sns.barplot(x=data.count()[:],y=data.count().index)
plt.xlabel('Non-Null Values Count')
plt.ylabel('Features')

data = data.drop([' _heatindexm',' _precipm',' _wgustm',' _windchillm'],axis=1)

#converting data into (yyyy-mm-dd HH:MM)
data['datetime_utc'] = pd.to_datetime(data['datetime_utc'].apply(lambda x: datetime.strptime(x,"%Y%m%d-%H:%M").strftime("%Y-%m-%d %H:%M")))
data['datetime_utc'].head()

data = data.set_index('datetime_utc',drop=True)
data.index.name = 'datetime'


fig, ax = plt.subplots()
data[' _tempm'].plot(figsize=(15,12),ax=ax)
ax.set_xlabel('Date-Time')
ax.set_ylabel('Temperature in C')
ax.set_title('Temperature in Delhi')
plt.show()

# Dropping the data before 1999
data = data['1999':]

print("Before : ", data.shape)
data.dropna(subset=[' _tempm'],inplace=True)
print("After :", data.shape)

data.index.minute.value_counts()
categoricalColumns = list(set(data.columns) - set(data._get_numeric_data().columns))
categoricalColumns

#missing value fillup by interpolation method
newdata = data.resample('H').mean().interpolate()
newdata.info()

#resampeling 
newdata[list(categoricalColumns)] = data[categoricalColumns].resample('H').first().ffill().head()
newdata.head()

def plotAggregateValues(data,column=None):
  if column in data.columns:
    plt.figure(figsize = (18,25))
    
    ax1 = plt.subplot(4,2,1)
    newdata[column].groupby(newdata.index.year).mean().plot(ax=ax1,title='yearly mean values')
    ax1.set_xlabel('years')
    ax1.set_ylabel(column)
  
    ax2 = plt.subplot(4,2,2)
    newdata[column].groupby(newdata.index.month).mean().plot(ax=ax2,title='monthly mean values')
    ax2.set_xlabel('months')
    ax2.set_ylabel(column)

     ax3 = plt.subplot(4,2,3)
     newdata[column].groupby(newdata.index.weekday).mean().plot(ax=ax3,title='weekdays mean values')
     ax3.set_xlabel('weekdays')
     ax3.set_ylabel(column)

    ax4 = plt.subplot(4,2,4)
    newdata[column].groupby(newdata.index.hour).mean().plot(ax=ax4,title='hourly mean values')
    ax4.set_xlabel('hours')
    ax4.set_ylabel(column)

     ax5 = plt.subplot(4,2,5)
     newdata[column].groupby(newdata.index.minute).mean().plot(ax=ax5,title='minute wise mean values')
     ax5.set_xlabel('minutes')
     ax5.set_ylabel(column)

     ax6 = plt.subplot(4,2,6)
     newdata[column].groupby(newdata.index.second).mean().plot(ax=ax6,title='seconds wise mean values')
     ax6.set_xlabel('seconds')
     ax6.set_ylabel(column)

  else:
    print("Column name not specified or Column not in the data")


plotAggregateValues(newdata,' _tempm')

def plotBoxNdendity(data,col=None):
  if col in data.columns:    
    plt.figure(figsize=(18,8))

    ax1 = plt.subplot(121)
    data.boxplot(col,ax=ax1)
    ax1.set_ylabel('Boxplot temperature levels in Delhi', fontsize=10)

    ax2 = plt.subplot(122)
    data[col].plot(ax=ax2,legend=True,kind='density')
    ax2.set_ylabel('Temperature distribution in Delhi', fontsize=10)

  else:
    print("Column not in the data")
plotBoxNdendity(data,' _tempm')

train = newdata[:'2016']
test = newdata['20120':]


#model_building

def decomposeNplot(data):
  decomposition = sm.tsa.seasonal_decompose(data)

  plt.figure(figsize=(15,16))

  ax1 = plt.subplot(411)
  decomposition.observed.plot(ax=ax1)
  ax1.set_ylabel('Observed')

  ax2 = plt.subplot(412)
  decomposition.trend.plot(ax=ax2)
  ax2.set_ylabel('Trend')

  ax3 = plt.subplot(413)
  decomposition.seasonal.plot(ax=ax3)
  ax3.set_ylabel('Seasonal')

  ax4 = plt.subplot(414)
  decomposition.resid.plot(ax=ax4)
  ax4.set_ylabel('Residuals')

  return decomposition


ftraindata = train[' _tempm'].resample('M').mean()
ftestdata = test[' _tempm'].resample('M').mean()

decomposition = decomposeNplot(ftraindata.diff(12).dropna())

results = adfuller(ftraindata.diff(12).dropna())
results

plt.figure(figsize=(10,8))

ax1 = plt.subplot(211)
acf = plot_acf(ftraindata.diff(12).dropna(),lags=30,ax=ax1)

ax2 = plt.subplot(212)
pacf = plot_pacf(ftraindata.diff(12).dropna(),lags=30,ax=ax2)

lags = [12*i for i in range(1,4)]

plt.figure(figsize=(10,8))

ax1 = plt.subplot(211)
acf = plot_acf(ftraindata.diff(12).dropna(),lags=lags,ax=ax1)

ax2 = plt.subplot(212)
pacf = plot_pacf(ftraindata.diff(12).dropna(),lags=lags,ax=ax2)


model = SARIMAX(ftraindata,order=(0,0,1),seasonal_order=(0,1,1,12),trend='n')
results = model.fit()
results.summary()

print(np.mean(np.abs(results.resid)))
diagnostics = results.plot_diagnostics(figsize=(10,10))

forecast = results.get_forecast(steps=len(ftestdata))

predictedmean = forecast.predicted_mean
bounds = forecast.conf_int()
lower_limit = bounds.iloc[:,0]
upper_limit = bounds.iloc[:,1]

plt.figure(figsize=(12,8))

plt.plot(ftraindata.index, ftraindata, label='train')
plt.plot(ftestdata.index,ftestdata,label='actual')

plt.plot(predictedmean.index, predictedmean, color='r', label='forecast')

plt.fill_between(lower_limit.index,lower_limit,upper_limit, color='pink')

plt.xlabel('Date')
plt.ylabel('Delhi Temperature')
plt.legend()
plt.show()

displayDirContent(base_dir)
filename = 'Sylhet.pkl'
joblib.dump(results,filename = data_dir + 'Models/' + filename)
