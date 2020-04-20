from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
import operator
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

def ReplaceNaN(X_train, y_train, X_test, y_test):

    #X_train_imp = X_train.replace([np.inf, -np.inf, np.nan], 0)
    #X_test_imp = X_test.replace([np.inf, -np.inf, np.nan], 0)

    X_train_imp = X_train.replace([np.inf, -np.inf], np.nan)    
    y_train_imp = y_train.drop(y_train.index[[X_train_imp[X_train_imp.isnull().any(axis=1)].index]])
    X_train_imp  = X_train_imp.dropna(how = 'any')

    X_test_imp = X_test.replace([np.inf, -np.inf], np.nan)    
    y_test_imp = y_test.drop(y_test.index[[X_test_imp[X_test_imp.isnull().any(axis=1)].index]])
    X_test_imp  = X_test_imp.dropna(how = 'any')

    #imp = Imputer()
    #X_train_imp = imp.fit_transform(X_train)
    #X_test_imp = imp.transform(X_test)
    return X_train_imp, y_train_imp, X_test_imp, y_test_imp

def Normalize(X_train, X_test):
    imp = MinMaxScaler()
    X_train_norm = imp.fit_transform(X_train)
    X_test_norm = imp.transform(X_test)
    return X_train_norm, X_test_norm


def RemoveOutliers(X_train, y_train, residual_threshold = 5):
    X = X_train.copy()
    y = y_train.copy()
    ransac = RANSACRegressor(LinearRegression(), 
                             max_trials=1000, 
                             min_samples=50, 
                             loss='squared_loss', 
                             residual_threshold=residual_threshold, 
                             random_state=0)


    ransac.fit(X, y)
    #print(ransac.inlier_mask_)
    return X[ransac.inlier_mask_], y[ransac.inlier_mask_]

def PCAComponents(X_train, explained_variance_threshold):    
    #Apply PCA by fitting the data and return the number of dimension
    #that provide explained variance according to the threshold parameter

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)

    pca = PCA().fit(X_train_norm)
    cumulative_variance = 0
    i = 0
    while cumulative_variance <= explained_variance_threshold:
        cumulative_variance += pca.explained_variance_ratio_[i]            
        i += 1
        
    return i


def FeatureImportance(X_train, y_train, n_epochs = 50, show_plot = False, max_size = 30):
 
    #remove null values
    #X_train = X_train.replace([np.inf, -np.inf, np.nan], 0)
    #X_train = X_train.dropna(how = 'any')
    
    total_importance = {}
    for epoch in range(1, n_epochs):

        #fit Random forest to identify most important features
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        feature_importance_dict = dict(zip(X_train.columns.tolist(), model.feature_importances_))
        sorted_top_features = sorted(feature_importance_dict.items(), key=operator.itemgetter(1), reverse=True)[:max_size]
        total_importance = dict(Counter(total_importance) + Counter({x[0]:x[1] for x in sorted_top_features}))
        
    sorted_total_importance = sorted(total_importance.items(), key=operator.itemgetter(1), reverse=True)[:max_size]
    feature_list = [x[0] for x in sorted_total_importance]

    if show_plot:
        objects = (feature_list)
        y_pos = np.arange(len(objects))
        performance = [10,8,6,4,2,1]
 
        plt.barh(y_pos, [x[1] for x in sorted_total_importance], align='center', alpha=0.5)
        plt.yticks(y_pos, objects)
        plt.xlabel('Usage')
        plt.title('Feature importance')
 
    
        plt.show()
    return feature_list
