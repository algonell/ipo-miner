# IPOMiner
Python utilities to predict future performance of upcoming [IPO (Initial Public Offering)](https://www.investopedia.com/terms/i/ipo.asp).

Checkout the [accompanying paper](https://github.com/algonell/IPOMiner/blob/master/NLP_ACL18.pdf) for more details.

<br/>

### What is this project?
This project is a collection of datasets and Python code to perform Text Mining on raw [SEC](https://www.sec.gov/ "Securities and Exchange Commission") [S-1 filings](https://www.investopedia.com/terms/s/sec-form-s-1.asp).  

<br/>

### What is the goal of this project?
The goal of this project is to apply Text Mining tools and techniques to spot investment opportunities in upcoming IPO. The system is comprised of three main modules. The first module is responsible for IPO data retrieval via [EDGAR SEC system](https://www.sec.gov/edgar/searchedgar/companysearch.html?). The second module is responsible for Text Mining. The third module is a classifier of upcoming IPO performance.  

<br/>

### How does it work?
Jupyter Notebooks are available for data retrieval, summarization, keywords extraction and Machine Learning.

__Start by running all cells in the following notebooks:__
- S-1 Downloader.ipynb - Download raw IPO data.
- Performance Downloader.ipynb - Download historical performance from Yahoo Finance.
- Summarizer.ipynb - Summarize raw S-1 filings.
- Keywords Extractor.ipynb - Extract keywords from S-1 filings.

__Then run all cells in the following notebooks:__
- 1 Baseline.ipynb - Transform raw IPO listings.
- 2 Sentiment Analysis.ipynb - Add Sentiment Analysis features.
- 3 Summarization.ipynb - Add summarization features.
- 4 Keywords.ipynb - Add keywords analysis.

__Making predictions:__
- Run all cells in Predictor.ipynb - Get upcoming IPO and predict performance.  

### Who will use this project?
This project is intended for traders and researchers as potential fork for alpha generation.

<br/>

# Directories
- Notebooks - Python scripts and Jupyter notebooks.
- Data - Raw S-1 SEC filings since 2000. Sample filings are provided.
- Datasets - CSV files used for training and evaluating Machine Learning models.
- Keywords - Top keywords for S-1 SEC filings.
- Summary - Summarized S-1 SEC filings.
