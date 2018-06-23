
# coding: utf-8

# # Keywords Extraction
# - Batch process

# In[4]:


#core
import pandas as pd

from bs4 import BeautifulSoup
from pathlib import Path

#NLP
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.summarization import keywords


# # Extract Keywords from IPO S-1 Filings

# In[5]:


#load previous progress
df = pd.read_csv('2 sentiment analysis.csv', index_col='Symbol')


# In[6]:


def extract_keywords(x):        
    #read S-1
    with open("./Data/" + x + ".htm", "r", encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html,"html5lib")
        text = soup.get_text(strip=True)

        #keywords
        words = keywords(text, ratio=0.01)

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
lemmatizer = WordNetLemmatizer()
counter = 0

for x in df.index:
    try:
        counter += 1
        print('\n( ' + str(counter) + ' / ' + str(df.shape[0]) + ' ) ' + str(x))

        #check if exists
        if Path("./Keywords/" + x + ".txt").is_file():
            print(x + ' data already exists, skipping...')
            continue

        #summarization and keywords
        extract_keywords(x)
    except Exception as e:
        print(x, e)
        Path('./Keywords/' + x + '.txt').touch()        

