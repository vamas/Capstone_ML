import pandas_datareader as pdr
import pandas_market_calendars as mcal
import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
from datetime import datetime, timedelta
import os.path
import urllib2
import pytz
from bs4 import BeautifulSoup
import errno, os, stat, shutil

    
class StockDataLoader():
    """ StockData class implements data loading and pre-processing logic """ 
    
    @property
    def raw_data_file(self):
        return self._raw_data_file
    
    @property
    def symbols(self):
        return self._symbols
    
    @property
    def short_term_window(self):
        return self._short_term_window
    
    @property
    def long_term_window(self):
        return self._long_term_window
    
    @property
    def start_date(self):
        return self._start_date
    
    @property
    def end_date(self):
        return self._end_date    
    
    @property
    def symbols(self):
        return self._symbols
    
    @property
    def finance_calendar(self):
        return self._finance_calendar
    
    @property 
    def raw_financial_data(self):
        return self._raw_financial_data
    
    @property
    def preprocessed_financial_data(self):
        return self._preprocessed_financial_data
    
    @property
    def all_tickers_dict(self):
        return self._all_tickers_dict
    
    @property
    def tickers_industry(self):
        return self._tickers_industry    
    
    def __init__(self, startdate, enddate, tickers = [], short_term_window = 15, long_term_window = 50):
        self.start_date = startdate
        self.end_date = enddate
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.raw_data_file =  os.path.join(os.getcwd(), 'SP500')
        self.preprocessed_financial_data = {}
        self.raw_financial_data = {}
        self.all_tickers_dict = self.CreateSP500Dict(tickers)
        #print(self.all_tickers_dict)

        
    def CreateSP500Dict(self, tickers):
        #### Section 1: Scrapes wikipedia page to get all tickers in the S&P 500

        thisurl = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" # the wikipedia url containing list of S&P 500 companies
        # it helps to visit the webpage and take a look at the source to understand
        #    how the html is parsed.

        myPage = urllib2.urlopen(thisurl) # opens this url

        mySoup = BeautifulSoup(myPage, "html.parser") # parse html soup 

        table = mySoup.find('table', {'class': 'wikitable sortable'}) # finds wiki sortable table in webpage html

        sector_tickers = dict() # create a dictionary to store all tickers according to sector
        for row in table.findAll('tr'): # find every row in the table
            col = row.findAll('td') # find every column in that row
            if len(col) > 0: # if there are columns in that row
                sector = str(col[3].string.strip()).lower().replace(' ', '_') # identify the sector in the row
                ticker = str(col[0].string.strip()) # identify the ticker in the row
                if sector not in sector_tickers: # if this sector is not a key in the dictionary
                    sector_tickers[sector] = list() # add this as a key to the dictionary
                if ticker in tickers or len(tickers) == 0:
                    sector_tickers[sector].append(ticker) # add the ticker to right key in the dictionary

        ticker_dict = {}
        for key in sector_tickers:
            values = sector_tickers[key]
            d = dict.fromkeys(values, key)    
            ticker_dict = merge_two_dicts(ticker_dict, d)

        self.tickers_industry = ticker_dict
 
        return sector_tickers  
    
    def GetTickersList(self):
        tickers = []
        for key in self.all_tickers_dict:
            value = self.all_tickers_dict[key]
            tickers = tickers + value     
        #remove exceptions
        #tickers.remove('BF.B')
        #tickers.remove('BRK.B')
        #add indices
        tickers.append('SPY')
        #tickers.append('DJI')
        #tickers.append('HSI')        
        #tickers.append('NYA')
        #tickers.append('TSE')
        #tickers.append('IXI')
        #tickers.append('CLF')
        return tickers
    
    def CreateFinanceCalendar(self):
        print("---Building calendar for the period from {0} to {1}".format(self.start_date, self.end_date))
        nyse = mcal.get_calendar('NYSE')
        early = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
        calendar_np = np.array(mcal.date_range(early, frequency='1D')  .map(lambda t: t.strftime('%Y-%m-%d')), dtype=object)
        self.finance_calendar = pd.DataFrame(data = calendar_np[0:],   
            index=calendar_np[0:],    
            columns=['Date'])        
        self.start_date = np.min(self.finance_calendar)
        self.end_date = np.max(self.finance_calendar)
        
        self.finance_calendar['Date_int'] = self.finance_calendar['Date'].map(lambda x: x.replace("-", ""))
        self.finance_calendar['Date_int'] = self.finance_calendar['Date_int'].astype(np.dtype('int32'))
        self.finance_calendar.set_index('Date_int', inplace = True)
        #del self.finance_calendar['Date']
        #self.finance_calendar.reset_index(inplace = True)
        ##del self.finance_calendar['index']
        #self.finance_calendar.reindex()
        #self.finance_calendar.reset_index(inplace = True)
        print("===Calendar is created")   
        
    def UpdateRawDataIncrementally(self):
        print("---Updating local data")
        #loop all tickers and populate missing data from Yahoo Finance!
        for symbol in self.GetTickersList():             
            #print("------Updating local data for {0}".format(symbol))
            start_date = datetime.strptime(self.start_date[0], "%Y-%m-%d")
            if symbol in self.raw_financial_data:
                #if dictionary already contains data for the symbol
                #check consistency and update delta
                max_stored_date = self.raw_financial_data[symbol].index.max()
                start_date = max_stored_date + timedelta(days=1)
            #Read Yeahoo data
            end_date = datetime.strptime(self.end_date[0], "%Y-%m-%d")            
            if start_date < end_date:
                try:
                    df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
                    if symbol in self.raw_financial_data:
                        df = self.raw_financial_data[symbol].append(df, ignore_index=False)
                        df.index = pd.to_datetime(df.index)
                    self.raw_financial_data[symbol] = df 
                except ValueError:
                    print("Yahoo Finance data could not be read for symbol {0}".format(symbol))
                    
        print("===Local data is updated")  
        
    def SaveRawData(self):
        print("---Saving data")
        for symbol in self.GetTickersList():
            file_name = os.path.join(self.raw_data_file, symbol + '.csv') 
            if os.path.isfile(file_name):
                os.remove(file_name)
            self.raw_financial_data[symbol].to_csv(path_or_buf=file_name)
        print("===Local data is saved")
                
    def GetRawDataFromFile(self):
        print("---Loading local data")
        
        #loop all tickers in S&P500 and load file if exists
        for symbol in self.GetTickersList():
            #print("------Loading local data for {0}".format(symbol))
            #file_name = self.raw_data_file + symbol + '.csv'
            file_name = os.path.join(self.raw_data_file, symbol + '.csv')
            if os.path.isfile(file_name):
                financial_data = pd.read_csv(file_name)
                financial_data.set_index('Date', inplace=True)
                financial_data.index = pd.to_datetime(financial_data.index)
                self.raw_financial_data[symbol] = financial_data      
        print("===Local data is loaded")  

           
    def LoadData(self, update_data_online = False):
        self.CreateFinanceCalendar()
        self.GetRawDataFromFile()
        if update_data_online:
            self.UpdateRawDataIncrementally()
            self.SaveRawData()
        self.OptimizeData()
        
    def OptimizeData(self):    
        #Optimize dataframe to take minimum RAM
        #Sort each dataframe and create integer index
        #Convert all TI except Volume to float16
        #Convert date to int
        print("---Optimizing")
        for key in self.raw_financial_data:
            self.raw_financial_data[key]['Token'] = key
            #add weekday            
            self.raw_financial_data[key].reset_index(inplace = True)
            self.raw_financial_data[key]['Weekday'] = self.raw_financial_data[key].Date.dt.dayofweek
            self.raw_financial_data[key]['Date'] = self.raw_financial_data[key].Date.astype(str)            
            self.raw_financial_data[key]['Date'] = self.raw_financial_data[key]['Date'].map(lambda x: x.replace("-", ""))
            self.raw_financial_data[key]['Date'] = self.raw_financial_data[key].Date.astype(np.dtype('int32'))   
            self.raw_financial_data[key]['Open'] = self.raw_financial_data[key].Open.astype(np.dtype('float64'))   
            self.raw_financial_data[key]['High'] = self.raw_financial_data[key].High.astype(np.dtype('float64'))   
            self.raw_financial_data[key]['Low'] = self.raw_financial_data[key].Low.astype(np.dtype('float64'))   
            self.raw_financial_data[key]['Close'] = self.raw_financial_data[key].Close.astype(np.dtype('float64'))   
            self.raw_financial_data[key]['Adj Close'] = self.raw_financial_data[key]['Adj Close'].astype(np.dtype('float64'))  
            
            #remove duplicates            
            self.raw_financial_data[key].drop_duplicates(['Date'], keep = 'last', inplace = True)
            self.raw_financial_data[key].set_index('Date', inplace = True)
                        
            #Align dataset with calendar
#            self.raw_financial_data[key] = pd.merge(self.finance_calendar, self.raw_financial_data[key], 
#                           how = 'inner', left_on = ['Date1'],  right_on = ['Date'])
#            del self.raw_financial_data[key]['index']
#            del self.raw_financial_data[key]['Date1']
            #print('{0}  {1}'.format(key, self.raw_financial_data[key]))
            
        print("===Optimizing is finished")
            

    def SelectSymbols(self, symbols):
        self.symbols = symbols
            
        
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z        
 
