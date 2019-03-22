# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:03:54 2019

@author: nEW u
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Opening the dataset
df=pd.read_csv(r'C:\Users\nEW u\Desktop\cse\Python\Data Science\3 months DS\Data-science-projects\Currency Exchange Rates\exchange_rate.csv')
y=list(df.head(0))[1:]
x=list(df['YEAR'])

df.plot(x,df[x])
