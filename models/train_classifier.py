import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from typing import Tuple, List
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pickle

def load_data(database_filepath: str)->Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    '''
    Function for loading the database into DataFrames
    Args: database_filepath: the path of the database
    Returns:    X: features (messages)
                y: categories
                An ordered list of categories
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM labeled_messages",engine) 

    # Features
    X = df['message']

    # Categories
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    categories = y.columns.values
    return X, y, categories

def tokenize(text:str)->List[str] :
    '''
    Function for tokenizing
    Args: Text string
    Returns: List of tokens
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]

def build_model()->GridSearchCV:
    '''
    Function for building Pipelines and GridSearch
    Args: None
    Returns: Random Forest Model
    '''
    # Pipeline for transforming data, fitting to model and predicting
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])
    
    parameters = {
        'clf__min_samples_split': [15,20],
        'clf__n_estimators': [100, 150,200],
        'clf__max_depth': [5, 8, 15, 25, 30],
        'clf__min_samples_leaf': [1, 2, 5, 10] }

    # GridSearch with the parameters
    cv = GridSearchCV(pipeline, param_grid=parameters,
                      scoring='accuracy',verbose= 1,n_jobs =-1)

    return pipeline

def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: List)->None:
    '''
    Function for evaluating model
    Args:   Model, features, labels, categories
    Returns: Classification report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for  idx, cat in enumerate(Y_test.columns.values):
        print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))

def save_model(model: GridSearchCV, model_filepath: str)-> None:
    '''
    Function for saving the model as picklefile
    Args: Model, filepath
    Returns: None
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Cargando data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Construyendo Modelo...')
        model = build_model()

        print('Entrenando Modelo...')
        model.fit(X_train, Y_train)

        print('Evaluando Modelo...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Grabando Modelo...\n    MODELO: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Modelo Guardado')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()