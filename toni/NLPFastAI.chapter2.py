# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:02:45 2021

author: trodriguez
"""

import spacy
#nlp = spacy.load('es_core_news_sm')
nlp = spacy.load('en_core_web_sm')
text = "Mary, don’t slap the green witch"
print([str(token) for token in nlp(text.lower())])