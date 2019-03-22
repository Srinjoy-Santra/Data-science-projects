# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:02:17 2019

@author: Srinjoy & Swayam
"""

import numpy as np
import pandas as pd

moviesdf=pd.read_csv(r'C:\Users\nEW u\Desktop\study\6th sem\Minor Project\ml-latest-small\movies.csv')
linksdf=pd.read_csv(r'C:\Users\nEW u\Desktop\study\6th sem\Minor Project\ml-latest-small\links.csv')
ratingsdf=pd.read_csv(r'C:\Users\nEW u\Desktop\study\6th sem\Minor Project\ml-latest-small\ratings.csv')

genre_list=moviesdf['genres']
genre_details=list()
for i in genre_list:
      genre_details.append(i.split('|'))
      
genre_details=[item for sublist in genre_details for item in sublist]       
print(list(set(genre_details)))
'''
df=pd.DataFrame(moviesdf, columns=moviesdf.columns[:23])
count=0
for i in genre_details:
      df[i]=0
for i in df['genres']:
      j=i.split('|')
      count=count+1
      for k in j:
            df[count,k]=1

print(df)
 '''     
