# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 01:23:58 2019

@author: nEW u
"""
from utilities import cleanText

def findBasics(name):
    linesList = cleanText(name + '.txt')
    
    #Check number of words
    wc=0
    for line in linesList:
        wc=wc+len(str(line).split(" "))
    print(wc)





filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat with Ipsita Haldar"
findBasics(filename)
