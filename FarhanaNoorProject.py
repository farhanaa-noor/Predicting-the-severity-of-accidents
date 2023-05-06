'''
Name: Farhana Ashrafi Noor
Email: farhana.noor20@myhunter.cuny.edu
Resources:
Title: Predicting the severity of accidents
URL: https://github.com/farhanaa-noor/Predicting-the-severity-of-accidents.git
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def data_cleaning(df):
    df.dropna(subset=['BOROUGH'], inplace=True)  # delete location with null values
    columns = ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4',
               'VEHICLE TYPE CODE 5']
    notnull_masks = [df[column_name].notnull() for column_name in columns]
    df['NUMBER OF CARS INVOLVED'] = sum(notnull_masks)  # new columns on number of vehicles involved in each accident

    df = df.drop(columns=['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4',
                          'VEHICLE TYPE CODE 5'], axis=1)
    df.dropna(subset=['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2',
              'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5'],
              how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def add_indicating_columns(df):
    # finding the unique values in each column
    set1 = set(df['CONTRIBUTING FACTOR VEHICLE 1'])
    set2 = set(df['CONTRIBUTING FACTOR VEHICLE 2'])
    set3 = set(df['CONTRIBUTING FACTOR VEHICLE 3'])
    set4 = set(df['CONTRIBUTING FACTOR VEHICLE 4'])
    set5 = set(df['CONTRIBUTING FACTOR VEHICLE 5'])
    # storing the unique values of all the contributing
    # factor in one place without any repetition
    new_set = set1 | set2 | set3 | set4 | set5
    for column in new_set:
        df[column] = [0] * df.shape[0]  # creating empty columns for each factor

    for s in new_set:
        print(s)
        mask = df.isin([s]).any(axis=1)  # Check if the value exists in any of the columns
        col_name = s  # Set a new column name for the indicating column
        df.loc[mask, col_name] = 1  # Set the indicating column to 1 if the value exists in any of the columns

    df = df.drop(columns=['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2',
                          'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4',
                          'CONTRIBUTING FACTOR VEHICLE 5'], axis=1)

    return df


def naive_bayes_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)


def random_forest_model(X, y):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    #print y pred
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(y_pred)
    print('Mean squared error: {:.2f}'.format(mse))
    print('R-squared score: {:.2f}'.format(r2))
    print("f1score ", f1)


def main():

    df = pd.read_csv('https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD', dtype={'ZIP CODE': str})

    df = df[['BOROUGH', 'CRASH TIME', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PERSONS INJURED', 'CONTRIBUTING FACTOR VEHICLE 1',
             'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4',
             'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3',
             'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']]

    df_clean = data_cleaning(df)
    print(df.shape[0])

    df_added_columns = add_indicating_columns(df_clean)
    df_added_columns.to_csv('modified_df.csv')

    df_added_columns['TOTAL PERSONS INVOLVED'] = df_added_columns['NUMBER OF PERSONS KILLED'] + df_added_columns['NUMBER OF PERSONS INJURED']

    df_added_columns['total_PAV'] = df_added_columns['TOTAL PERSONS INVOLVED'] + df_added_columns['NUMBER OF CARS INVOLVED']

    # find mean normalized
    mean = df_added_columns['total_PAV'].mean()
    std = df_added_columns['total_PAV'].std()
    df_added_columns['PAV_norm'] = (df_added_columns['total_PAV']-mean)/std
    df_added_columns['PAV_normalized'] = (df_added_columns['PAV_norm'] - df_added_columns['PAV_norm'].min()) / \
                                         (df_added_columns['PAV_norm'].max()-df_added_columns['PAV_norm'].min())

    df_added_columns = df_added_columns.drop(columns=['total_PAV', 'BOROUGH', 'PAV_norm'])
    df_added_columns= df_added_columns.dropna()

    df_added_columns = df_added_columns.sort_values(by=['PAV_normalized'])

    pers15 = np.percentile(df_added_columns['PAV_normalized'], 15)
    pers50 = np.percentile(df_added_columns['PAV_normalized'], 50)
    pers90 = np.percentile(df_added_columns['PAV_normalized'], 90)

    df_added_columns['severity_category'] = df_added_columns['PAV_normalized'].apply(lambda x: 'low' if x <= pers15 else ('moderate' if pers15 < x <= pers90 else 'high'))

    df_added_columns['severity_category'] = df_added_columns['severity_category'].astype('category')
    cat_columns = df_added_columns.select_dtypes(['category']).columns
    df_added_columns[cat_columns] = df_added_columns[cat_columns].apply(lambda x: x.cat.codes)
    print(df_added_columns[cat_columns].value_counts())
    y = df_added_columns['severity_category']
    x = df_added_columns.drop(columns=['severity_category'])
    random_forest_model(x, y)


if __name__ == "__main__":
    main()

