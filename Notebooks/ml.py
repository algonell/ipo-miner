#Machine Learning utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn import preprocessing
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve

from sklearn import ensemble
import xgboost as xgb

def standardize(df):
    '''Returns standardized DataFrame'''
	
    return (df-df.mean())/df.std()
	
def encode(df, col):
    '''Returns encoded Series'''	
    
    le = preprocessing.LabelEncoder()
    return le.fit_transform(df[col])	
	
def run_ml_flow(df):
    '''Runs Machine Learning flow, returns evaluation DataFrame'''
    
    targets = ['1D', '1W', '1M', '3M']
    evaluation = pd.DataFrame(columns=targets, index=pd.MultiIndex.from_product([['AUC', 'f1', 'log loss'], ['LR', 'RF']]))

    for target in targets:

        #split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(df.values[:,:-4], df[target].map(lambda x: 1 if x > 0 else 0).values, test_size=0.2, shuffle=False)

        #classifiers
        clfs = {
            'RF' : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1),
            'LR' : LogisticRegression(random_state=1),
            #'Vote' : VotingClassifier(estimators=[('lr', LogisticRegression(random_state=1)), ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1))], voting='soft')
        }

        #fit
        for k, clf in clfs.items():
            clf.fit(X_train, y_train)

        #evaluate
        if target == '3M':
            plt.figure(figsize=(15, 5))
        
        for k, clf in clfs.items():
            predictions = clf.predict(X_test)
            probas = clf.predict_proba(X_test)

            evaluation.loc['AUC', k] = roc_auc_score(y_test, predictions)
            evaluation.loc['f1', k] = f1_score(y_test, predictions)
            evaluation.loc['log loss', k] = log_loss(y_test, probas)
			
            if target == '3M':
                plot_roc(y_test, probas[:,1], k)
                plot_log_loss(y_test, probas[:,1], k)
        
        if target == '3M':
            plt.show()		
    
    return evaluation

def rank_features_xgb(X, Y, columns):
    # fit 
    model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.01, seed=0)
    model.fit(X, Y)   

    #list feature importance
    #model.get_booster().feature_names = columns
    imp_dict = model.get_booster().get_fscore()
    imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1), reverse=True)

    imp_arr = np.asarray(imp_dict)
    indices = np.empty([imp_arr.shape[0], 0])

    for x in imp_arr[:,0]:
        indices = np.append(indices, x.replace('f', ''))

    indices = indices.astype(int)
    importances = imp_arr[:,1].astype(int)    

    # Print the feature ranking
    #print("Feature ranking:")

    #for f in range(indices.shape[0] - 1):
    #    print(f, 'index', indices[f], X.columns[indices[f]], importances[f]    )

    # Plot the feature importances of the forest
    plt.figure(figsize=(15,5))
    plt.title("XGB Feature importances")
    plt.bar(range(indices.shape[0]), importances[:indices.shape[0]], color="r", align="center")
    plt.xticks(range(X.shape[1] + 1), np.array(columns)[indices], rotation='vertical')
    plt.xlim([-1, X.shape[1] + 1])
    plt.show()         

    # plot via xgb
    #fig, ax = plt.subplots(1,1,figsize=(15,10))
    #xgb.plot_importance(model, ax=ax)     
    
    return indices

def rank_features_etc(X, Y, columns):
    # supervised ranking
    model = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=0)
    model.fit(X, Y)
    
    #list feature importance
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    #print("Feature ranking:")

    #for f in range(X.shape[1] - 1):
    #    print(f, 'index', indices[f], X.columns[indices[f]], importances[indices[f]])
        
    # Plot the feature importances of the forest
    plt.figure(figsize=(15,5))
    plt.title("ETC Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), np.array(columns)[indices], rotation='vertical')
    plt.xlim([-1, X.shape[1] + 1])
    plt.show()
    
    return indices	
	
def show_feature_importance(df, target_col):
    '''Shows feature importances by ETC and XGC'''
    X, Y = df.values[:,:-4], df[target_col].map(lambda x: 1 if x > 0 else 0).values
    indices_sklearn = rank_features_etc(X, Y, df.columns[:-4])
    rank_features_xgb(X, Y, df.columns[:-4])	
	
def plot_roc(y_test, probas, k):
    '''
	Plots ROC curve
	
	y_test -- predictions
	probas -- probabilities
	k -- classifier
	'''
    
    plt.subplot(1, 2, 1)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.plot(fpr, tpr, label=k)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='best')
	
def plot_log_loss(y_test, probas, k):
    '''
	Plots log loss for instances
	
	y_test -- predictions
	probas -- probabilities
	k -- classifier	
	'''
	
    #collect log loss for instances
    plt.subplot(1, 2, 2)
    v = []

    for x, y in zip(y_test,probas):
        v.append(ll(x, y))

    v = np.sort(v)    

    #plot log loss for instances
    plt.plot(v, label=k)
    plt.axhline(y=log_loss([1, 0], [0.5, 0.5]), color='black', linestyle='--')
    plt.xlabel('Instances')
    plt.ylabel('Log Loss')
    plt.title('Log Loss')
    plt.legend(loc='best')

def ll(yt, yp):
    '''
    returns logarithmic loss
    
    yt -- prediction
    yp -- probability
    '''
    
    return -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))	