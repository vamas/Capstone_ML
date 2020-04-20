###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
from scipy.interpolate import spline

def evaluateClassifiers(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 1
    colors = ['#A00000','#00A0A0','#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):

                # Creative plot code
                ax[j/3, j%3].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])
                #ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45])
                #ax[j/3, j%3].set_xticklabels(["1%", "10%", "100%"])
                #ax[j/3, j%3].set_xlabel("Training Set Size")
                #ax[j/3, j%3].set_xlim((-0.1, 3.0))
                
                

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()

def featureImportance(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    feat_labels = X_train.columns[:]
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[[f]]]))

    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]),
           importances[indices],
           color='lightblue',
           align='center')
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

def evaluateClassifierPrediction(results, classifier_name, metric):
    acc_test_values = []
    acc_train_values = []
    labels = []

    for key in results:
        acc_test_values.append(array([results[key][n][classifier_name][metric + '_train'] for n in results[key]]).mean())
        acc_train_values.append(array([results[key][n][classifier_name][metric + '_test'] for n in results[key]]).mean())
        labels.append(key)

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(acc_test_values) + 1), acc_test_values)
    plt.plot(range(1, len(acc_train_values) + 1), acc_train_values)
    ax = plt.axes()
    ax.set_ylim([0, 1])
    ax.set_xticks(range(1, len(acc_test_values) + 1))
    plt.xlabel('Token')
    ax.grid()
    plt.legend(['Test','Train'], loc=4)
    plt.title('Accuracy')
    plt.show()
    print(labels)

def ResultOutput(dataframe):
    df = dataframe.copy()
    idx = df.groupby(['Ticker'])['R2 test'].transform(max) == df['R2 test']
    df = df[idx]
    print(df)

def PlotR2Score(df, title, score): 
    # data to plot
    n_groups = 5
    score_AAPL = df[(df['Ticker'] == 'AAPL')]['R2 test'].as_matrix().tolist()
    score_MSFT = df[(df['Ticker'] == 'MSFT')]['R2 test'].as_matrix().tolist()
    score_ACN = df[(df['Ticker'] == 'ACN')]['R2 test'].as_matrix().tolist()
    score_GOOG = df[(df['Ticker'] == 'GOOG')]['R2 test'].as_matrix().tolist()
    score_CSCO = df[(df['Ticker'] == 'CSCO')]['R2 test'].as_matrix().tolist()
    score_EBAY = df[(df['Ticker'] == 'EBAY')]['R2 test'].as_matrix().tolist()
    score_EA = df[(df['Ticker'] == 'EA')]['R2 test'].as_matrix().tolist()
    score_HP = df[(df['Ticker'] == 'HP')]['R2 test'].as_matrix().tolist()
    score_IBM = df[(df['Ticker'] == 'IBM')]['R2 test'].as_matrix().tolist()
    score_INTC = df[(df['Ticker'] == 'INTC')]['R2 test'].as_matrix().tolist()

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.05
    opacity = 0.8

    rectsAAPL = plt.bar(index, score_AAPL, bar_width, alpha=opacity,  label='AAPL') 
    rectsMSFT = plt.bar(index + bar_width, score_MSFT, bar_width, alpha=opacity, label='MSFT')
    rectsACN = plt.bar(index + 2 * bar_width, score_ACN, bar_width, alpha=opacity, label='ACN')
    rectsGOOG = plt.bar(index + 3 * bar_width, score_GOOG, bar_width, alpha=opacity, label='GOOG')
    rectsCSCO = plt.bar(index + 4 * bar_width, score_CSCO, bar_width, alpha=opacity, label='CSCO')
    rectsEBAY = plt.bar(index + 5 * bar_width, score_EBAY, bar_width, alpha=opacity, label='EBAY')
    rectsEA = plt.bar(index + 6 * bar_width, score_EA, bar_width, alpha=opacity, label='EA')
    rectsHP = plt.bar(index + 7 * bar_width, score_HP, bar_width, alpha=opacity, label='HP')
    rectsIBM = plt.bar(index + 8 * bar_width, score_IBM, bar_width, alpha=opacity, label='IBM')
    rectsINTC = plt.bar(index + 9 * bar_width, score_INTC, bar_width, alpha=opacity, label='INTC')

    legend = ax.legend(loc='lower center', bbox_to_anchor=(1.1, 0.2), shadow=True)
    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(index + bar_width, ('SVR', 'Ada', 'Lasso', 'Ridge', 'Linear'))
    if score == 'mse':
        plt.ylim(-0.25, np.max(df['Score'].as_matrix()) + 1)        
    else:
        plt.ylim(-0.25, 1)        
    plt.tight_layout()
    plt.show()
    
    

    



def ResisualPlot(y_train, y_train_pred, y_test, y_test_pred):
    plt.scatter(y_train_pred,  
                y_train_pred - y_train, 
                c='steelblue',
                edgecolor='white',
                marker='o', 
                s=35,
                alpha=0.9,
                label='training data')
    plt.scatter(y_test_pred,  
                y_test_pred - y_test, 
                c='limegreen',
                edgecolor='white',
                marker='s', 
                s=35,
                alpha=0.9,
                label='test data')

    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
    plt.xlim([-10, 50])
    plt.tight_layout()

    # plt.savefig('images/10_14.png', dpi=300)
    plt.show()

    
 