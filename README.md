# IPOMiner
Python utilities to predict future performance of upcoming [IPO](https://www.google.com "Initial Public Offering).

### What is this project?
This project is a collection of datasets and Python code to perform Text Mining on raw [SEC](https://www.sec.gov/ "Securities and Exchange Commission") [S-1 filings](https://www.investopedia.com/terms/s/sec-form-s-1.asp).

### How does it work?
Python scripts are available for data retrieval, summarization, keywords extraction and Machine Learning.
```python
#training (since)
python train.py '2000-01-01'

#prediction
python predict.py 'Company Name'
```

### What is the goal of this project?
The goal of this project is to apply Text Mining tools and techniques to spot investment opportunities in upcoming IPO. The system is comprised of three main modules. The first module is responsible for IPO data retrieval via [EDGAR SEC system](https://www.sec.gov/edgar/searchedgar/companysearch.html?). The second module is responsible for Text Mining. The third module is a classifier of upcoming IPO performance.

### Who will use this project?
This project is intended for traders and researchers as potential fork for alpha generation.

# Directories
- Notebooks - Python scripts and Jupyter notebooks.
- Data - Raw S-1 SEC filings since 2000. Sample filings are provided. Full data (zipped ~650MB/~5.5GB, +2400 filings) is available at: https://goo.gl/Wnx1Dt
- Datasets - CSV files used for training and evaluating Machine Learning models.
- Keywords - Top keywords for S-1 SEC filings.
- Summary - Summarized S-1 SEC filings.
