# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 01:00:29 2019

@author: nEW u
"""
import sys
import re
import matplotlib.pyplot as plt
import nltk
from utilities import cleanText
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer() # Our Great Sentiment Analyzer

def analyze(name):
    linesList = cleanText(name + '.txt')
    neutral, negative, positive = 0, 0, 0

    for index, sentence in enumerate(linesList):
        print("Processing {0}%".format(str((index * 100) / len(linesList))))
       
        # Ignore Emoji
        if re.match(r'^[\w]', sentence):
            continue
       
        scores = sentiment_analyzer.polarity_scores(sentence)
       
        # We don't need that component
        scores.pop('compound', None)
       
        maxAttribute = max(scores, key=lambda k: scores[k])

        if maxAttribute == "neu":
            neutral += 1
        elif maxAttribute == "neg":
            negative += 1
        else:
            positive += 1

    total = neutral + negative + positive
    print("Negative: {0}% | Neutral: {1}% | Positive: {2}%".format(
        negative*100/total, neutral*100/total, positive*100/total))
   
    
    labels = 'Neutral', 'Negative', 'Positive'
    sizes = [neutral, negative, positive]
    colors = ['#66c5f4', '#f47469', '#8cf442']

    # Plot
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)

    plt.axis('equal')
    plt.title("Sentiment Analysis")
    plt.show()


filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat(1)"
analyze(filename)
filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat(2)"
analyze(filename)
￼￼filename = r"C:\Users\nEW u\Downloads\WhatsApp Chat(3)"
analyze(filename)