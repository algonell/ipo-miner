
# coding: utf-8

# # Summarization
# - Summarize S-1 raw filings
# - Batch process

# In[1]:


#core
import pandas as pd

from bs4 import BeautifulSoup
from pathlib import Path

#NLP
from gensim.summarization import summarize


# # Summarize IPO S-1 Filings

# In[2]:


#load previous progress
df = pd.read_csv('2 sentiment analysis.csv', index_col='Symbol')


# In[3]:


def summarize_symbol(x):
    #read S-1
    with open("./Data/" + x + ".htm", "r", encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html,"html5lib")
        text = soup.get_text(strip=True)

        #summary and keywords
        summary = summarize(text[:250000], ratio=0.01)
        
        #write summary
        with open("./Summary/" + x + ".txt", "w", encoding="utf-8") as f:
            f.write(summary)    


# In[ ]:


#batch process
counter = 0

for x in df.index:
    try:
        counter += 1
        print('\n( ' + str(counter) + ' / ' + str(df.shape[0]) + ' ) ' + str(x))

        #check if exists
        if Path("./Summary/" + x + ".txt").is_file():
            print(x + ' data already exists, skipping...')
            continue

        #summarization and keywords
        summarize_symbol(x)
    except Exception as e:
        print(x, e)
        #for now expection are only too large files that cause out of memory
        #anyway it's imposible to summarize so then write empty files for later jobs
        Path('./Summary/' + x + '.txt').touch()      

