# TITLE: Detecting Crypto Market Crashes

## Proposer: Nathaniel Merrill

## PROBLEM STATEMENT: 
Cryptocurrency is a new and exciting way for young people to make large returns from
their investments, but it is also a highly volatile market where your investment could shrink
to zero overnight. In this project, we will be investigating some different factors
that affect a crypto market crash using historical market data and comments from Reddit.

## DELIVERABLE: 
We will start by trying to correlate the occurrence different keywords from Reddit posts under
r/CryptoCurrency using the [PRAW webscraper](https://praw.readthedocs.io) to rises or 
falls in some popular cryptocurrencies (Bitcoin, ETH, DOGE, etc.) and create some
plots. Then, if we find some correlations, and time permits, we can work to build a small 
neural network using PyTorch to predict market crashes (i.e. a fall of a certain percent in a day) 
using simple features such as the number of occurences of the name of the cryptocurrency 
in r/CryptoCurrency on that day along with a small chunk of historical market data.

## SKILLS NEEDED: 
Python programming and the ability to create a good presentation are the only requirements.

## DATA: 
- Historical stock data: [https://finance.yahoo.com](https://finance.yahoo.com)
- Reddit data comes from PRAW linked above.

## AUTHOR: 
Nathaniel Merrill (nmerrill@udel.edu)
