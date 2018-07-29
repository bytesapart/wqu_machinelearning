import warnings
from urllib2 import urlopen
import datetime
from lxml import etree
import sqlite3
import pandas as pd
import sys
from pandas_datareader import data as pdr  # The pandas Data Module used for fetching data from a Data Source
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fix_yahoo_finance import pdr_override  # For overriding Pandas DataFrame Reader not connecting to YF

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def yahoo_finance_bridge():
    """
    This function fixes problems w.r.t. fetching data from Yahoo Finance
    :return: None
    """
    pdr_override()


def obtain_parse_wiki_snp500():
    # stores the current time
    page = urlopen(url)
    htmlparser = etree.HTMLParser()
    tree = etree.parse(page, htmlparser)
    symbolslist = tree.xpath('//table[1]/tbody/tr')[1:]

    symbols = []
    for symbol in symbolslist:
        tds = symbol.getchildren()
        sd = {'ticker': tds[0].getchildren()[0].text,
              'name': tds[1].getchildren()[0].text,
              'sector': tds[3].text}
        symbols.append((sd['ticker'], sd['name'], sd['sector']))

    return symbols


if __name__ == '__main__':
    # Get a list of tickers from Wikipedia by scraping
    list_of_tickers_from_wiki = obtain_parse_wiki_snp500()
    # Make a DataFrame from it and write it to a SQL file
    try:
        conn = sqlite3.connect('StockDB.sqlite3')
        ticker_df = pd.DataFrame(list_of_tickers_from_wiki, columns=['Ticker', 'Name', 'Sector'])
        # Save to SQLite3 DB
        ticker_df.to_sql('Companies', conn, index=False, if_exists='replace')
    finally:
        conn.close()

    # Create the Yahoo Finance Bridge so that you get the data
    yahoo_finance_bridge()
    start = datetime.datetime.now()
    dled_data = pdr.get_data_yahoo(ticker_df['Ticker'].values.tolist(), start=start - datetime.timedelta(1), end=start, auto_adjust=True)
    # dled_data = pd.read_pickle('dled_data.p')
    # Get data from Yahoo Finance, w.r.t. price for each ticker
    for index, row in ticker_df.iterrows():
        ticker_df.loc[index, 'Price'] = dled_data['Close'][row['Ticker']][0]
        ticker_df.loc[index, 'Volume'] = dled_data['Volume'][row['Ticker']][0]
        ticker_df.loc[index, 'Date'] = start.strftime('%d-%b-%Y')

    # Save it to Database
    try:
        conn = sqlite3.connect('StockDB.sqlite3')
        ticker_df.to_sql('TickerWithData', conn, if_exists='replace', index=False)
    finally:
        conn.close()
    sys.exit()
