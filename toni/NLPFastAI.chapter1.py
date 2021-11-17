# -*- coding: utf-8 -*-
"""
Created on Sat May  1 13:15:39 2021

author: trodriguez
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ['Time flies flies like an arrow.',
          'Fruit flies like a banana.']


"""
One-hot-encoding Representation
"""

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
vocab = [*one_hot_vectorizer.vocabulary_]
sns.heatmap(one_hot, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence1','Sentence 2'])

plt.show()

"""
TF-IDF Representation
"""

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
vocab = [*tfidf_vectorizer.vocabulary_]
sns.heatmap(tfidf, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence1','Sentence 2'])
plt.show()