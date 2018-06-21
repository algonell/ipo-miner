#Machine Learning utilities
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

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
    evaluation = pd.DataFrame(columns=['AUC', 'f1', 'log loss'])

    for target in targets:

        #split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(df.values[:,:-4], df[target].map(lambda x: 1 if x > 0 else 0).values, test_size=0.2, shuffle=False)
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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
        for k, clf in clfs.items():
            #print(k)
            predictions = clf.predict(X_test)
            probas = clf.predict_proba(X_test)

            auc = roc_auc_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            ll = log_loss(y_test, probas)
            #print('AUC:', auc)
            #print('f1:', f1)
            #print('log loss:', ll)
            #print(confusion_matrix(y_test, predictions))
            #print('\n')

        #save
        evaluation.loc[target] = [auc, f1, ll]
    
    return evaluation.T	