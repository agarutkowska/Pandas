#!/usr/bin/env python
# coding: utf-8

# In[59]:


# Wczytanie potrzebnych bibliotek
import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import math


# In[53]:


confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv") 
recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")


# In[54]:


confirmed = confirmed.set_index(["Province/State","Country/Region","Lat","Long"])
recovered = recovered.set_index(["Province/State","Country/Region","Lat","Long"])
deaths = deaths.set_index(["Province/State","Country/Region","Lat","Long"])


# In[55]:


confirmed = confirmed.stack()
recovered = recovered.stack()
deaths = deaths.stack()


# In[56]:


total = pd.DataFrame({"c": confirmed, "r": recovered, "d": deaths})


# In[57]:


total = total.reset_index()


# In[58]:


total = total.assign(time=pd.to_datetime(total['level_4']))


# In[60]:


total = total.assign(time_from_1march=(total['time'] - pd.Timestamp('2020/03/01')) / np.timedelta64(1, 'D'))


# In[62]:


poland = total[total["Country/Region"]=='Poland']

plt.subplot(2,1,1)
x = poland['time_from_1march']
y = poland['c']
plt.plot(x, y, '.') # potwierdzone przypadki dzienne 
plt.subplot(2,1,2)
plt.plot(x[1:], np.diff(y), '.') # przyrosty dzienne


# In[63]:


def gompertz(t, N0, a, c):
    return N0 * np.exp(-c * (np.exp(-a * t)))


# In[64]:


popt, pcov = scipy.optimize.curve_fit(gompertz, x.values, y.values, p0=[1.,1.,1.])


# In[67]:


def diff_fun(fun, h=1e-7):
    return lambda x : (fun(x + h) - fun(x - h)) / 2 / h

t = np.linspace(0, 120, 121)
ym = gompertz(t, *popt)
plt.subplot(2,1,1)
plt.plot(t, ym, "g-")
plt.plot(x, y, ".")

plt.subplot(2,1,2)
plt.plot(x[1:], np.diff(y),'k.')
first_der = diff_fun(lambda x: gompertz(x, *popt))(t)
plt.plot(t, first_der, 'r-')


# In[69]:


total_cases = popt[0]
print(f'Przewidywana liczba zachorowań na COVID-19 w Polsce to {total_cases: 0.00f}.')

t_max = t[np.argmax(first_der)]
print(f'Model przewiduje szczyt epidemii na {t_max: 0.0f} dzień od 1. marca.')

t_end = scipy.optimize.brenth(lambda t: gompertz(t, *popt) - 0.99*popt[0], a = 0, b = 1000)
print(f'Model przewiduje koniec epidemii na {t_end: 0.0f} dzień od 1. marca.')




