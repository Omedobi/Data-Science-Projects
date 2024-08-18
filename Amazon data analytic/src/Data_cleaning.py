import pandas as pd
import numpy as np
import logging
import hashlib , itertools

from sklearn.impute import SimpleImputer

class DataHandling:
    def __init__(self, dataframe):
        """
        Initialize the cleaner with a pandas Dataframe.
        """
        self.df = dataframe
        
    def capitalize_columns(self):
        """
        Capitalize the first letter of each colum name.
        """
        try:
            self.df.columns = self.df.columns.str.capitalize()
        except Exception as e:
            logging.error(f'Error in capitalizing the first letter: {e}')
            raise e
        
    def strip_columns(self):
        """remove spaces from the column names."""
        try:
            self.df.columns = self.df.columns.str.strip()
        except Exception as e:
            logging.error(f'Error in removing spaces: {e}')
            raise e    
        
    def clean_numeric_column(self, column_name, remove_symbol = None):
        """
        Clean a numeric column by removing a specified symbol (like currency or percentage) and converting to float.
        
        Parameters:
        - column_name: str, the name of the column to clean.
        - remove_symbol: str, the symbol to remove from the column values (e.g., '₹' for currency or '%' for percentages).
        """
        try:
            if remove_symbol:
                self.df[column_name] = self.df[column_name].str.replace(remove_symbol, '', regex=False).str.replace(',', '').astype(float)
            else:
                self.df[column_name] = self.df[column_name].str.replace(',', '').astype(float)
        except Exception as e:
            logging.error(f'Error in removing the symbol  ₹ or % from: {e}')
            raise e
                
    def convert_to_float(self, *column_names):
        """convert selected columns to float.
        """
        try:
            for column_name in column_names:
                #remove both '|' and ',' character using regex
                self.df[column_name] = self.df[column_name].str.replace('[|,]', '', regex=True)
                
                self.df[column_name] = self.df[column_name].replace('',np.nan)
                # convert to float
                self.df[column_name] = pd.to_numeric(self.df[column_name], errors='coerce')
        except Exception as e:
            logging.error(f'Error in converting selected column {column_name} to float: {e}')
            raise e
                
    def apply_cleaning(self):
        """Apply all the cleaning steps to the DataFrame"""
        try:
            self.capitalize_columns()
            self.strip_columns()
            self.clean_numeric_column('Actual_price', '₹')
            self.clean_numeric_column('Discounted_price', '₹')
            self.clean_numeric_column('Discount_percentage', '%')
            self.convert_to_float('Rating','Rating_count')
        except Exception as e:
            logging.error(f'Error in applying cleaning steps: {e}')
            raise e
        
    def get_cleaned_dataframe(self):
        """Return the cleaned dataframe."""
        return self.df
    


class CategoricalHandling:
    def __init__(self, dataframe):
        self.df = dataframe
        
    def clean_and_process_column(self, column):
        """Clean and process a single column in the dataframe.
        Splits the values based on delimiters, cleans them, and generates new IDs if necessary.
        """
        try:
            # Split the column into a list based on the delimiters ',' or '|'
            self.df[column] = self.df[column].str.split('[,|]', regex=True)
            
            def clean_and_generate(values):
                cleaned_values = []
                for value in values:
                    # Clean the value by stripping whitespace and converting to lowercase
                    cleaned_data = value.strip().lower().replace('[^a-zA-Z0-9]', '', regex=True)
                    # Generate a hash-based ID
                    generate_id = int(hashlib.md5(cleaned_data.encode()).hexdigest(), 16) % 10**8
                    cleaned_values.append(f'{cleaned_data}:{generate_id}')
                return cleaned_values              
            # Apply the cleaning and processing to each list in the column
            self.df[column] = self.df[column].apply(lambda x: clean_and_generate(x) if isinstance(x, list) else x)
        
        except Exception as e:
            logging.error(f'Error in cleaning and processing column {column}: {e}')
            raise e
    
    def process_dataframe(self):
        """Process the entire dataframe by applying the cleaning and processing to each categorical feature."""
        try:
            for column in self.df.columns:
                if self.df[column].dtype == 'object':
                    self.clean_and_process_column(column)
            return self.df
        except Exception as e:
            logging.error(f'Error in applying the process to the dataframe: {e}')
            raise e
            
    def flatten_column(self, column):
        """Flatten the lists in a column into a single list."""
        try:
            # Flatten the list of lists in the column into a single list
            flat_list = list(itertools.chain.from_iterable(self.df[column].dropna()))
            return flat_list
        except Exception as e:
            logging.error(f'Error in flattening column {column}: {e}')
            raise e
            
    def apply_cat(self, func, columns=None):
        """Apply a custom function to the entire DataFrame or to specific columns.
        
        Parameters:
        - func: function, the custom function to apply.
        - columns: list of str or None, the columns to apply the function to. If None, applies to the entire DataFrame.
        """
        try:
            if columns:
                self.df[columns] = self.df[columns].apply(func)
            else:
                self.df = self.df.apply(func)
            return self.df
        except Exception as e:
            logging.error(f'Error in applying function to the DataFrame: {e}')
            raise e