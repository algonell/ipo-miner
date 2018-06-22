
# coding: utf-8

# # Summarization
# - Summarize S-1 raw filings
# - Extract keywords
# - Batch process

# In[1]:


#core
import pandas as pd

from bs4 import BeautifulSoup
from pathlib import Path

#NLP
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.summarization import summarize
from gensim.summarization import keywords


# # Summarize IPO S-1 Filings

# In[2]:


#load previous progress
df = pd.read_csv('2 sentiment analysis.csv', index_col='Symbol')


# In[ ]:


def summarize_and_extract_keywords(x):
    '''
    x -- symbol
    '''
        
    #read S-1
    with open("./Data/" + x + ".htm", "r", encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html,"html5lib")
        text = soup.get_text(strip=True)

        #summary and keywords
        lemmatizer = WordNetLemmatizer()
        summary = summarize(text, ratio=0.01)
        words = keywords(text, ratio=0.01)
        
        #write summary
        with open("./Summary/" + x + ".txt", "w", encoding="utf-8") as f:
            f.write(summary)

        #write keywords
        with open("./Keywords/" + x + ".txt", "w", encoding="utf-8") as f:
            #lemmatize
            lemmatized_keywords = []

            for w in word_tokenize(words):
                lemmatized_keywords.append(lemmatizer.lemmatize(w))

            lemmatized_keywords = list(set(lemmatized_keywords))

            #write
            for k in lemmatized_keywords:
                f.write('%s\n' % k)     


# In[ ]:


#batch process
counter = 0

for x in df.index:
    try:
        counter += 1
        print('\n( ' + str(counter) + ' / ' + str(df.shape[0]) + ' ) ' + str(x))

        #check if exists
        if Path("./Summary/" + x + ".txt").is_file() and Path("./Keywords/" + x + ".txt").is_file():
            print(x + ' data already exists, skipping...')
            continue

        #summarization and keywords
        summarize_and_extract_keywords(x)
    except Exception as e:
        print(x, e)

