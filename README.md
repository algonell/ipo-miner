# IPOMiner
Python utilities to predict future performance of upcoming IPO.

### What is this project?
This project is a collection of datasets and Python code to perform Text Mining On raw SEC S-1 filings.

### How does it work?
Python scripts are available for data retrieval, summarization, keywords extraction and Machine Learning.
```python
#training (since)
python train.py '2000-01-01'

#prediction
python predict.py 'Company Name'
```

### What is the goal of this project?
The goal of this project is to apply Text Mining tools and techniques to spot investment opportunities in upcoming IPO (Initial Public Offering). The system is comprised of three main modules. The first module is responsible for IPO data retrieval via EDGAR SEC system. The second module is responsible for Text Mining. The third module is a real-time classifier of upcoming IPO performance.

### Who will use this project?
This project is intended for traders and researchers as potential fork for alpha generation.

# Directories
- Notebooks - Python scripts and Jupyter notebooks.
- Data - Raw S-1 SEC filings since 2000. Sample filings are provided. Full data (zipped ~650MB/~5.5GB, +2400 filings) is available at: https://goo.gl/Wnx1Dt
- Datasets - CSV files used for training and evaluating Machine Learning models.
- Keywords - Top keywords for S-1 SEC filings.
- Summary - Summarized S-1 SEC filings.
