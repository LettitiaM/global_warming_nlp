from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier



def splitDataset(df):
    # Seperate features and tagret variables
    X = df['tokens']
    y = df['sentiment']


    # Split the dataset into train & validation (25%) for model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def modelTrain(model, string, X_train, y_train):
    modelTrain = Pipeline([('tfidf',TfidfVectorizer()),
               ('clf',model)])
    #modelTrain = model

    modelTrain.fit(X_train, y_train)

    # save the model to disk
    filename = 'plk_objects/'+string + '_model.pkl'
    pickle.dump(modelTrain, open(filename, 'wb'))
    
def loadModel(string):
    # load the decision model from disk
    filename = 'plk_objects/'+string + '_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    #result = model.score(X_test, y_test)
    #print(result)
    
    return model

def getPrediction(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def modelEvaluation(y_test, y_pred):
    # Generate a classification Report for the random forest model
    print(metrics.classification_report(y_test, y_pred))

    # Generate a normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / cm.sum(axis=1).reshape(-1,1)