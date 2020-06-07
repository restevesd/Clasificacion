# Disaster Response Udacity Pipeline

[Figure Eight](https://www.figure-eight.com/) is a company that provides datasets for data analysis and delivered us a dataset with messages classified into different categories to analyze emergency response messages.

Using Machine Learning we will be able to predict the category of the message


### The process for this exercise is as follows:

#### 1. Data Cleaning and Processing
Clean the data so that it can be used in a Machine Learning model (https://github.com/restevesd/Clasificacion/blob/master/data/process_data.py).
    
#### 2. Trainign Model
We use a pipeline to automate tasks and a prediction model is made.
See script in (https://github.com/restevesd/Clasificacion/blob/master/models/train_classifier.py).

#### 3. Model in production
Execute intructions below.
        
### Instructions:
#### Run the following instructions in the directory root directory of the project to set up all assets:

To run ETL:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

To run ML pipeline:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### Run the following command in the app's directory to run your web app:

`python run.py`

#### Go to http://0.0.0.0:3001/


### File structure of project:

1.  ../app - folder for web app

    ../app/run.py - flask web app
    
    ../templates - .html templates
    

2.  ../data - folder for files for the datasets

    ../data/disaster_categories.csv - raw file containing the categories
    
    ../data/disaster_messages.csv - raw file containing the messages
    
    ../data/process_data.py
    
    ../data/DisasterResponse.db - database for the clean data
    

3.  ../models - folder for the classifier model and pickle file

    ../models/train_classifier.py - model training script
    
    ../models/classifier.pkl - saved model when running `python train_classifier.py`
    

4.  README.md
