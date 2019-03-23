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
    print("Word count=",wc)
    
   

filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat(1)"
findBasics(filename)
filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat(2)""
findBasics(filename)
￼￼filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat(3)"
findBasics(filename)