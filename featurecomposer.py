import pandas as pd
import numpy as np
from scipy.stats import zscore
import os.path
import errno, os, stat, shutil
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from preprocessor import PCAComponents
from sklearn.decomposition import PCA
import math
import copy

class FeatureComposer():
    
    def __init__(self, stockdata_dict, indicators, pack_size, injecting_tokens, pred_indicator, base_indicator = 'Adj Close',
                 apply_zscore = True, export_datasets = False, short_term_window = 15, long_term_window = 30, log_transform = False,
                 base_features_absolute = True):

        self.stockdata_dict = copy.deepcopy(stockdata_dict)
        self.indicators = indicators
        self.pack_size = pack_size
        self.injecting_tokens = injecting_tokens
        self.apply_zscore = apply_zscore
        self.export_datasets = export_datasets
        self.pred_indicator = pred_indicator
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.base_indicator = base_indicator
        self.log_transform = log_transform
        self.base_features_absolute = base_features_absolute

    def Compose(self):
        '''
        Extract features
        '''
        print('Composer - Extracting features')
        stockDataPreprocessor = FeatureExtractor(dataframes = self.stockdata_dict, short_term_window = self.short_term_window, 
                                                 long_term_window = self.long_term_window, log_transform = self.log_transform,
                                                 base_features_absolute = self.base_features_absolute)
        self.stockdata_dict = stockDataPreprocessor.Extract(self.base_indicator, self.indicators, self.pred_indicator)
        self.indicators = stockDataPreprocessor.GetIndicators()
        
        '''
        Pack features
        '''
        print('Composer - Packing features')
        stockDataPacker = FeaturePack(stockdata_dict = self.stockdata_dict, indicators = self.indicators, pack_size = self.pack_size)
        stockDataPacker.PackFeatures()

        '''
        Inject other stocks
        '''
        print('Composer - Injecting features')
        token_data_list = []
        i = 1
        #pack stock data that is to be added to target stock
        if len(self.injecting_tokens) > 0:
            for key in self.injecting_tokens:
                df = self.stockdata_dict[key].copy()
                self.alter_dataframe_column_names(df, str(self.injecting_tokens[i-1]))
                token_data_list.append(df)
                i += 1    
            alter_data = pd.concat(token_data_list, join = 'inner', axis = 1)   

            #add injecting data and apply zscore
            for key in self.stockdata_dict:
                self.stockdata_dict[key] = pd.concat((self.stockdata_dict[key], alter_data), join = 'inner', axis = 1)
        
        '''
        Concat individual stock datasets into single dataframe
        '''
        print('Composer - Combining datasets')
        data_list = []
        for key in self.stockdata_dict:
            df = self.stockdata_dict[key]
            df['token'] = key
            df['date'] = df.index
            data_list.append(df)

        print('Composer - DONE')

    def GetData(self, ticker, predicting_day_index = 1, feature_list = [], remove_NaN = True, predict_change = False):
        df = self.stockdata_dict[ticker].copy()
        del df['token']
        if remove_NaN:
            df = df.replace([np.inf, -np.inf], np.nan)    
            df  = df.dropna(how = 'any')
        df = df.reset_index(drop = True)
        date_index_df = df[['date']]
        del df['date']        
        if not feature_list:
            features_df = df[[c for c in df.columns.tolist() if c != 'Label']]
        else: 
            features_df = df[[c for c in feature_list if c != 'Label']]
        if predict_change:
            #labels = pd.Series(((df['Label'].shift(-predicting_day_index) - df['Label'])/df['Label']*100.0).round(2))
            labels = pd.Series(((df['Label'].shift(-predicting_day_index) - df['Label'])).round(2))
        else:
            labels = pd.Series((df['Label'].shift(-predicting_day_index)))
        return features_df, labels, pd.Series(date_index_df['date'])

   
    def alter_dataframe_column_names(self, df, alter_str = ''):
        old_columns = df.columns
        new_columns = dict([i, alter_str + '|' + i] for i in old_columns)
        df.rename(columns = new_columns, inplace = True)  

    def GetSampleBackwards(self, ticker, training_set_end_date, training_set_size, predicting_day_index = 1, feature_list = []):
        features, labels, date_indexes = self.GetData(ticker, predicting_day_index = predicting_day_index, feature_list = feature_list)
        training_set_end_index = date_indexes[date_indexes == training_set_end_date].index[0]
        training_start_date = date_indexes[training_set_end_index - training_set_size]
        training_end_date = date_indexes[training_set_end_index]
        testing_start_date =  date_indexes[training_set_end_index + predicting_day_index]
        testing_end_date =  date_indexes[training_set_end_index + predicting_day_index]
        training_data_indexes = date_indexes[(date_indexes >= training_start_date) \
                                                        & (date_indexes <= training_end_date)].index.tolist()
        testing_data_indexes = date_indexes[(date_indexes >= testing_start_date) \
                                                       & (date_indexes <= testing_end_date)].index.tolist()

        return [features.iloc[training_data_indexes], labels[training_data_indexes], 
                features.iloc[testing_data_indexes], labels[testing_data_indexes], date_indexes]

    def GetSample(self, ticker, start_index, training_size, testings_split_coef, predicting_day_index = 1, feature_list = [], predict_change = False):        
        features, labels, date_indexes = self.GetData(ticker, predicting_day_index = predicting_day_index, feature_list = feature_list, 
                                                      predict_change = predict_change)
        training_start_date = date_indexes[start_index]
        training_end_date = date_indexes[start_index + training_size - 1]
        testing_start_date = date_indexes[start_index + training_size]
        testing_end_date = date_indexes[start_index + training_size + int(math.ceil(training_size * testings_split_coef))]

        training_data_indexes = date_indexes[(date_indexes >= training_start_date) \
                                                        & (date_indexes <= training_end_date)].index.tolist()
        testing_data_indexes = date_indexes[(date_indexes >= testing_start_date) \
                                                       & (date_indexes <= testing_end_date)].index.tolist()

        return [features.iloc[training_data_indexes], labels[training_data_indexes], 
                features.iloc[testing_data_indexes], labels[testing_data_indexes], date_indexes]

    def Predict(self, ticker, prediction_date, predicting_day_index, training_set_size, regressor):
        features, labels, date_indexes = self.GetData(ticker, predicting_day_index)
        prediction_date_index = date_indexes[date_indexes == prediction_date].index[0]
        predicting_date = date_indexes[prediction_date_index + predicting_day_index]
        training_end_date = prediction_date
        training_start_date = date_indexes[prediction_date_index - training_set_size]        
        testing_start_date = predicting_date
        testing_end_date = predicting_date
        training_data_indexes = date_indexes[(date_indexes >= training_start_date) \
                                                        & (date_indexes <= training_end_date)].index.tolist()
        testing_data_indexes = date_indexes[(date_indexes >= testing_start_date) \
                                                        & (date_indexes <= testing_end_date)].index.tolist()

        X_train = features.iloc[training_data_indexes]
        y_train = labels[training_data_indexes] 
        X_test = features.iloc[testing_data_indexes]
        y_test = labels[testing_data_indexes] 

        pipe_lr = Pipeline([
                        ('nrm', MinMaxScaler()),
                        ('pca', PCA(n_components = PCAComponents(X_train, 0.9))), 
                        ('rgr', regressor)
                       ])

        pipe_lr.fit(X_train, y_train)
        prediction = pipe_lr.predict(X_test)

        return [ticker, predicting_date, prediction, y_test]
        
    
class FeatureExtractor():
        
    def __init__(self, dataframes, short_term_window, long_term_window, log_transform, base_features_absolute):
        self.dataframes = dataframes
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.log_transform = log_transform
        self.base_features_absolute  = base_features_absolute
        
    def GetIndicators(self):
        return self.indicators

    def LogTransform(self, df):
        if self.log_transform:
            df = df[[s for s in df.columns.tolist() if s != 'Weekday' and s != 'Token' and s != 'Label']].apply(lambda x: np.sign(x) * np.log(np.absolute(x) + 1))    
       
    def CreateIndicatorDynamic(self, df):
        #for column in [s for s in df.columns.tolist() if 'Low' in s or 'Open' in s or 'Close' in s or 'High' in s or 'Adj Close' in s \
        #        or 'TEMA' in s or 'EMA' in s or 'MA' in s or 'Volume' in s]:
        for column in [s for s in df.columns.tolist() if s != 'Weekday' and s != 'Token' and s != 'Label']:
            df[column] = pd.Series((((df[column] - df[column].shift(1)))/df[column].shift(1) * 100.0).round(4))
        return df

    def Extract(self, source_feature, indicators, pred_indicator):
        '''List of indicators [EMA, MA, BOLL, ROC, WILLR, MOM]'''        
        #loop over dictinary and apply preprocessing routines
        for key in self.dataframes:
            df = self.dataframes[key]
            
            '''Add custom TIs'''
            if 'BOLLH' in indicators:
                df['BOLLH'] = self.CalcBollingerBand(df[source_feature], low = False)
            if 'BOLLL' in indicators:
                df['BOLLL'] = self.CalcBollingerBand(df[source_feature], low = True)
            if 'ROC' in indicators:
                df['ROC'] = self.CalcRateOfChange(df[source_feature], self.short_term_window)
            if 'WILLR' in indicators:
                df['WILLR'] = pd.Series((df['High'] - df['Close'])/(df['High'] - df['Low']) * 100.0)
            if 'MOM' in indicators:
                df['MOM'] = self.CalcMomentum(df[source_feature], self.short_term_window)
            if 'RSI' in indicators:
                df['RSI'] = self.CalcRSI(df[source_feature], self.short_term_window)
            if 'CCI' in indicators:
                df['CCI'] = self.CalcCCI(df, self.short_term_window)
            if 'TEMA' in indicators:
                df['TEMA'] = self.CalcTEMA(df[source_feature], self.short_term_window)
            if 'VIX' in indicators:
                df['VIX'] = self.CalcVIX(df[source_feature], self.short_term_window)
            if 'BIAS' in indicators:
                df['BIAS'] = self.CalcBIAS(df[source_feature], self.short_term_window)
            if 'HighLowDiff' in indicators:
                df['HighLowDiff'] = pd.Series(df['High'] - df['Low'])
            if 'CloseOpenDiff' in indicators:
                df['CloseOpenDiff'] = pd.Series(df['Close'] - df['Open'])
            if 'EMA' in indicators:
                df['EMA'] = self.CalcEMA(df[source_feature], self.short_term_window)
            if 'EMA_COEFF' in indicators:
                EMAl = self.CalcEMA(df[source_feature], self.long_term_window)
                EMAs = self.CalcMovingAvg(df[source_feature], self.short_term_window)
                df['EMA_COEFF'] = EMAl /  EMAs 
            if 'MA' in indicators:
                df['MA'] = self.CalcMovingAvg(df[source_feature], self.short_term_window)
            if 'D_MA5' in indicators and 'D_MA10' in indicators:
                df['D_MA5'] = pd.Series(df['MA'] - df['MA'].shift(5))
                df['D_MA10'] = pd.Series(df['MA'] - df['MA'].shift(10))
            if 'D_TEMA5' in indicators and 'D_TEMA10' in indicators:
                df['D_TEMA5'] = pd.Series(df['TEMA'] - df['TEMA'].shift(5))
                df['D_TEMA10'] = pd.Series(df['TEMA'] - df['TEMA'].shift(10))
                df['D_TEMA20'] = pd.Series(df['TEMA'] - df['TEMA'].shift(20))
                df['D_TEMA30'] = pd.Series(df['TEMA'] - df['TEMA'].shift(30))
            if 'D_VIX5' in indicators and 'D_VIX10' in indicators:
                df['D_VIX5'] = pd.Series(df['VIX'] - df['VIX'].shift(5))
                df['D_VIX10'] = pd.Series(df['VIX'] - df['VIX'].shift(10))
            if 'D_RSI5' in indicators and 'D_RSI10' in indicators:
                df['D_RSI5'] = pd.Series(df['RSI'] - df['RSI'].shift(5))
                df['D_RSI10'] = pd.Series(df['RSI'] - df['RSI'].shift(10))
            if 'D_BIAS5' in indicators and 'D_BIAS10' in indicators:
                df['D_BIAS5'] = pd.Series(df['BIAS'] - df['BIAS'].shift(5))
                df['D_BIAS10'] = pd.Series(df['BIAS'] - df['BIAS'].shift(10))      
                #df['D_BIAS20'] = pd.Series(df['BIAS'] - df['BIAS'].shift(20))    
                #df['D_BIAS30'] = pd.Series(df['BIAS'] - df['BIAS'].shift(30))    
            if 'D_EMA5' in indicators and 'D_EMA10' in indicators:
                df['D_EMA5'] = pd.Series(df['EMA'] - df['EMA'].shift(5))
                df['D_EMA10'] = pd.Series(df['EMA'] - df['EMA'].shift(10))
            if 'D_Adj Close5' in indicators and 'D_Adj Close10' in indicators:
                df['D_Adj Close5'] = pd.Series(df['Adj Close'] - df['Adj Close'].shift(5))
                df['D_Adj Close10'] = pd.Series(df['Adj Close'] - df['Adj Close'].shift(10))
            if 'D_CloseOpenDiff5' in indicators and 'D_CloseOpenDiff10' in indicators:
                df['D_CloseOpenDiff5'] = pd.Series(df['CloseOpenDiff'] - df['CloseOpenDiff'].shift(5))
                df['D_CloseOpenDiff10'] = pd.Series(df['CloseOpenDiff'] - df['CloseOpenDiff'].shift(10))
            if 'D_HighLowDiff5' in indicators and 'D_HighLowDiff10' in indicators:
                df['D_HighLowDiff5'] = pd.Series(df['HighLowDiff'] - df['HighLowDiff'].shift(5))
                df['D_HighLowDiff10'] = pd.Series(df['HighLowDiff'] - df['HighLowDiff'].shift(10))
            
            '''Add predicting indicator'''
            df['Label'] = pd.Series(df[pred_indicator])
    
            '''Convert features from absolute to change over time values'''
            if not self.base_features_absolute:
                df = self.CreateIndicatorDynamic(df)
                            
            '''Apply log transformation'''
            self.LogTransform(df)

        self.indicators = [s for s in df.columns.tolist() if s != 'Token' and s != 'Label']       

        return self.dataframes

    def CalcBIAS(self, values, window):
        EMA = values.ewm(span = window).mean()
        BIAS = (values - EMA) / EMA * 100
        return BIAS

    def CalcTEMA(self, values, window):
        EMA1 = self.CalcEMA(values, window)
        EMA2 = self.CalcEMA(EMA1, window)
        EMA3 = self.CalcEMA(EMA2, window)
        TEMA = 3*EMA1 - 3*EMA2 + EMA3
        return TEMA

    def CalcCCI(self, df, window):
        '''Calculate commodity channel index'''
        TP = (df['High'] + df['Low'] + df['Close']) / 3 
        CCI = pd.Series((TP - pd.rolling_mean(TP, window)) / (0.015 * pd.rolling_std(TP, window)))
        return CCI

    def CalcRSI(self, values, window):
        '''Calculate Relative Strenght Index
        '''
        series = pd.Series(values - values.shift(1))
        dUp, dDown = series.copy(), series.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = pd.rolling_mean(dUp, window)
        RolDown = pd.rolling_mean(dDown, window).abs()
        return 100 - 100 / (1 + RolUp / RolDown)

    def CalcEMA(self, values, window):
        '''Expotentia moving average'''
        ema = values.ewm(span = window).mean()
        return ema

    def CalcMovingAvg(self, values, window):
        '''Simple moving average'''
        sma = values.rolling(window=window, center=False).mean()
        return sma    

    def CalcBollingerBand(self, values, low = False):
        '''Bollinger bands'''
        stdev = np.std(values)
        boll_band = values.copy()
        if low:
            boll_band -= 2 * stdev
        else:
            boll_band += 2 * stdev
        return boll_band

    def CalcRateOfChange(self, values, window):
        '''Rate of change'''
        roc = values[window:] / values.shift(window) * 100.0
        return roc

    def CalcMomentum(self, values, window):
        '''Rate of change'''
        mom = values[window:] - values.shift(window)
        return mom

    def CalcVIX(self, values, window):
        '''VOLATILITY  index'''
        IR = pd.Series((values - values.shift(1)) - 1)
        STD_IR = pd.rolling_std(IR, window)
        VIX = pd.Series(STD_IR * math.sqrt(252))
        return  VIX

class FeaturePack():
    
    def __init__(self, stockdata_dict, indicators, pack_size):
        self.stockdata_dict = stockdata_dict
        self.indicators = indicators
        self.pack_size = pack_size
        self.feature_set_prefix = 'DAY'
        
    def GetFeatureSetPrefix(self, feature, index):        
        return self.feature_set_prefix + str(index) + '|' + feature
    
    def GetFeatureSetIndex(self, feature_set_prefix):
        return find_between(feature_set_prefix, self.feature_set_prefix, '|')
        
    def GetFeatureSetFeature(self, feature_set_prefix):
        return find_between(feature_set_prefix + '|', '|', '|')
        
    def PackFeatures(self):
        for key in self.stockdata_dict:            
            for day in range(0, self.pack_size):
                for indicator in self.indicators:
                    self.stockdata_dict[key][self.GetFeatureSetPrefix(indicator, day)] = pd.Series(self.stockdata_dict[key][indicator].shift(day))
            #remove columns that don't have prefix
            self.stockdata_dict[key] = self.stockdata_dict[key].drop([s for s in self.stockdata_dict[key].columns.tolist() if self.feature_set_prefix not in s and s != 'Label'], axis=1)
        return self.stockdata_dict     
    

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
        
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)