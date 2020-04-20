# Machine Learning Nanodegree Capstone Project

In this Python project I'm using using supervised regression learning algorithms to predict future prices for the given period. Source data used in the project are publicly available S&P500 companies’ historical stock prices dataset that can be obtained from Yahoo! Finance web-service. Detailed implementation is described in CapstoneProject_Implementation.pdf file.

## Documentation

### Project folder structure:
```
root
    |__Output - folder is used for output data dumps
    |__Debug - folder is used for debugging dumps
    |__SP500 - folder where stock data is stored
```

### Classes
```
stockdataloader.py - stock data uploader
featurecomposer.py - main class that responsible for preparing dataset
preprocessor.py - class that contains preprocessing routines, normalization etc.
stockdatavisual.py - class that contains various visualizations for the project
helpers.py - other routines
```

### Thirdt-Party libraries
```
pip install pandas_datareader 
pip install pandas_market_calendars 
pip install fix_yahoo_finance 
pip install beautifulsoup4
pip install pytz
```

## License
DupDetectorML is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Feel free to do whatever you want with it.