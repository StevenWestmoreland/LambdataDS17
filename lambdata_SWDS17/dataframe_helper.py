'''
Some functions to help cleaning and handling dataframes
'''

import pandas as pd
import numpy as numpy
from sklearn.model_selection import train_test_split
from pdb import set_trace as breakpoint
from IPython.display import display
from lambdata_SWDS17.dataframe_helper import report_missing_values


class My_Data_Splitter():
    def __init__(self, df, features, target):
        self.df = df
        self.target = target
        self.X = df[features]
        self.y = df[target]

    def train_validation_test_splitter(self, train_size=0.7,
                                       val_size=0.1, test_size=0.2,
                                       random_state=None, shuffle=True):
        '''
        This function is a utility wrapper around the Scikit-learn train_test_split that splits arrays or
        matrices into train, validation, and test subsets.
        Args:
            X (Numpy array or Dataframe): This is the first param.
            y (Numpy array or Dataframe): This is the second param.
            train_size (float or int): Proportion of dataset to include in train split (0 to 1).
            val_size (float or int): Proportion of dataset to include in validation split(0 to 1).
            test_size (float or int): Propoertion of dataset to include in test split (0 to 1).
            random_state (int): Controls shuffling applied to data before applying split for reproducibility.
            shuffle (bool): Whether or not to shuffle data before splitting.
        Returns:
            Train, test, and validation dataframes for features (X) and target (y).
        '''
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
            random_state=random_state, shuffle=shuffle
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def date_divider(self, date_col):
        '''
        This function splits an existing datatime object into Year, Month, and Day features.
        Args:
            df (Numpy array or Dataframe): Object where the date_column is located.
            date_col (string): Name of the date_column to be split within the passed dataframe.
        Returns:
            Modified dataframe with the new Year, Month, and Day columns attached to the end.
        '''
        converted_df = self.df.copy()
        converted_df["Year"] = pd.to_datetime(converted_df[date_col]).dt.year
        converted_df["Month"] = pd.to_datetime(converted_df[date_col]).dt.month
        converted_df["Day"] = pd.to_datetime(converted_df[date_col]).dt.day
        return converted_df

    def print_split_summary(self, X_train, X_val, X_test):
        print('######################## TRAINING DATA ########################')
        print(f'X_train Shape: {X_train.shape}')
        display(X_train.describe(include='all').transpose())
        print('')
        print('######################## VALIDATION DATA ######################')
        print(f'X_val Shape: {X_val.shape}')
        display(X_val.describe(include='all').transpose())
        print('')
        print('######################## TEST DATA ############################')
        print(f'X_test Shape: {X_test.shape}')
        display(X_test.describe(include='all').transpose())
        print('')


class State_Standardization():
    def __init__(self, df):
        self.df = df
    
    def full_to_abbreviated(stateFullName):
        '''
        This function takes a state's full name and returns the abbreviation.
        Args:
            stateFullName (String): The full state name.
        Returns:
            The state abbreviation.
        '''
        full_to_abbr = {'Alabama': 'AL', 'Alaska': 'AK', 'American Samoa': 'AS',
        'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
        'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
        'Florida': 'FL', 'Georgia': 'GA', 'Guam': 'GU', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'National': 'NA',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
        'North Dakota': 'ND', 'Northern Mariana Islands': 'MP', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
        }
        return full_to_abbr.get(stateFullName.title())
    
    def abbreviated_to_full(AB):
        '''
        This function takes a state abbreviation and returns the full state name.
        Args:
            AB (String): The state abbreviation.
        Returns:
            The full state name.
        '''
        abbr_to_full = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas',
        'AS': 'American Samoa', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado',
        'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'GU': 'Guam', 'HI': 'Hawaii',
        'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts',
        'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota',
        'MO': 'Missouri', 'MP': 'Northern Mariana Islands', 'MS': 'Mississippi',
        'MT': 'Montana', 'NA': 'National', 'NC': 'North Carolina', 'ND': 'North Dakota',
        'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
        'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
        'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island',
        'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
        'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin Islands', 'VT': 'Vermont',
        'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
        }
        return abbr_to_full.get(AB.upper())

def report_missing_values(df):
    '''Print a pretty report of missing values'''
    print('Need to write code here.')
