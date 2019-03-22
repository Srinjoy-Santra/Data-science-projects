# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:54:03 2019

@author: Srinjoy Santra
"""

import pandas as pd

gscore = pd.read_csv(r'C:\Users\nEW u\Desktop\study\6th sem\Minor Project\ml-latest\genome-scores.csv')
gtags = pd.read_csv(r'C:\Users\nEW u\Desktop\study\6th sem\Minor Project\ml-latest\genome-tags.csv')


result = pd.merge(gscore,gtags,on='tagId',how='outer')

result.to_csv(r'C:\Users\nEW u\Desktop\study\6th sem\Minor Project\ml-latest\combined_genom.csv')
