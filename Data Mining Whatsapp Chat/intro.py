# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:49:14 2019

@author: Srinjoy Santra
"""
filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat with Ipsita Haldar.txt"
import re

mediaPattern = r"(\<Media omitted\>)" # Because it serves no purpose
regexMedia = re.compile(mediaPattern, flags=re.M)

dateAndTimepattern = r"(\d+\/\d+\/\d+)(,)(\s)(\d+:\d+)(\s)(\w+)(\s)(-)(\s\w+)*(:)"
regexDate = re.compile(dateAndTimepattern, flags=re.M)

def cleanText(filename):    
    chat = open(filename,encoding="latin-1")
    chatText = chat.read()
    chat.close()

    # 01/09/17, 11:34 PM - Amfa:

    """
    Removes the matches and
    replace them with an empty string
    """
    chatText = regexMedia.sub("", chatText)
    chatText = regexDate.sub("", chatText)

    lines = []

    for line in chatText.splitlines():
        if line.strip() is not "": # If it's empty, we don't need it
            lines.append(line.strip())

    return lines


