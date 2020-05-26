import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads Message and Categories
    Args:
        messages_filepath: The path of the messages csv
        categories_filepath: The path of the categories cv

    Returns:
        df (pandas dataframe): The combined messages and categories df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merge = pd.merge(messages, categories, on='id') 
    return merge

def clean_data(df):
    """Cleans the data:
        - Eliminate duplicates
        - Eliminate messages missing classes
        - Cleans categories column

    Args:
        df (dataframe): join categories and messages

    Returns:
        df (pandas dataframe): Cleaned dataframe and categories
    """
    # categories column
    categories = df.categories.str.split(';', expand=True)
    row = categories[:1]

    # categories names
    category_colname = row.applymap(lambda s: s[:-2]).iloc[0, :].tolist()
    categories.columns = category_colname

    # get each value as integer
    categories = categories.applymap(lambda s: int(s[-1]))

    # add to original dataframe the categories 
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # clean
    df.drop_duplicates(subset='message', inplace=True)
    df.dropna(subset=category_colnames, inplace=True)
    df.related.replace(2, 0, inplace=True)

    return df

def save_data(df, database_filename):
    """Saves the results to a sqlite db
    Args:
        df (dataframe): The dataframe clean
        database_filename (string): the file of the db

    Returns:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('mensajes_etiquetados', engine, index=False, if_exists='replace')
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Cargando Data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Limpiando Data...')
        df = clean_data(df)

        print('Guardando Data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Data guardada en base!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
