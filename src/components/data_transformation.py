# import os
# import sys
# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from category_encoders import TargetEncoder
# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import FunctionTransformer

# # Custom transformer to calculate 'daysUsed'
# class DaysUsedTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, date_column, reg_column):
#         self.date_column = date_column
#         self.reg_column = reg_column
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         X_copy = X.copy()
#         X_copy['daysUsed'] = (pd.to_datetime(X_copy[self.date_column]).dt.year - X_copy[self.reg_column])
#         return X_copy.drop(columns=[self.date_column, self.reg_column])

# class RemoveZerosTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns):
#         self.columns = columns
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         return X[(X[self.columns] != 0).all(axis=1)]

# class DataTransformationConfig:
#     def __init__(self):
#         self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# def yearOfRegistration_filter_transform(X):
#     return (X[(X > 1940) & (X < 2020)])


# def kilometer_filter_transform(X):
#     return (X[(X > 5000) & (X < 150000)])


# def powerPS_filter_transform(X):
#     return (X[(X > 100) & (X < 2500)])

# def remove_andere_filter_transform(X):
#     return (X[X != 'andere'])

# def year_used_filter_transform(Y):
#     Z = (2016 - Y)
#     return pd.DataFrame(Z, columns=['yearUsed'])



# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
#         self.target_encoder = TargetEncoder()
#         self.scaler = StandardScaler()

#     def list2int(list):
#         return int(list[0]), int(list[1])
    
    

#     def get_data_transformer_object(self):
#         '''
#         This function is responsible for data transformation
#         '''
#         try:
            
#             preprocessing_one = ColumnTransformer(
#                 [
#                     ("drop_cols", 'drop', tuple(['seller', 'name', 'index', 'offerType', 'monthOfRegistration', 'nrOfPictures', 'postalCode', 'dateCreated', 'dateCrawled','lastSeen'])),
#                     # logging.info('Dropping of the columns is done.')
#                     # ('remove_zeros', remove_zeros_transformer, tuple(['powerPS', 'kilometer']))
#                     # # ('price_filter', 'passthrough',(['price'], lambda x: (x > 100) & (x < 60000))),
#                     ('powerPS_filter', FunctionTransformer(powerPS_filter_transform), ['powerPS']),
#                     ('kilometer_filter', FunctionTransformer(kilometer_filter_transform), ['kilometer']),
#                     ('yearOfRegistration', FunctionTransformer(yearOfRegistration_filter_transform),['yearOfRegistration']),
#                     ('remove_andere_from_brand', FunctionTransformer(remove_andere_filter_transform), ['brand']),
#                     ('remove_andere_from_model', FunctionTransformer(remove_andere_filter_transform), ['model']),
#                     ('remove_andere_from_fuelType', FunctionTransformer(remove_andere_filter_transform), ['fuelType']),
#                 ], remainder='passthrough'
#             )

#             feature_engineering = ColumnTransformer(
#                 [
#                     ('year_used', FunctionTransformer(year_used_filter_transform), ['yearOfRegistration'])
#                 ]
#             )

#             preprocessing_two = ColumnTransformer(
#                 [
#                     ('drop_yearOfRegistration', 'drop', tuple(['yearOfRegistration'])),
#                     ('target_encode', self.target_encoder, tuple(['fuelType', 'abtest', 'vehicleType', 'gearbox', 'brand', 'model', 'notRepairedDamage'])),
#                     ('standardize', self.scaler, tuple(['abtest', 'vehicleType', 'gearbox', 'powerPS','brand', 'model', 'kilometer', 'fuelType', 'notRepairedDamage', 'yearUsed']))
#                 ], remainder='passthrough'
#             )
#                     # ('days_used', 'passthrough', DaysUsedTransformer(date_column='lastSeen', reg_column='yearOfRegistration'), tuple(['lastSeen', 'yearOfRegistration']))
#                     # # ('scaler', StandardScaler(), (['powerPS', 'kilometer'])),
                    
#                     # ('move_column', 'passthrough', tuple(['abtest', 'vehicleType', 'gearbox', 'powerPS', 'brand', 'model', 'kilometer', 'fuelType', 'notRepairedDamage', 'daysUsed']))

#                     # ("drop_cols", 'drop', ['seller', 'name', 'index', 'offerType', 'monthOfRegistration', 'nrOfPictures', 'postalCode', 'dateCreated', 'dateCrawled']),
#                     # ('remove_zeros', remove_zeros_transformer, ['powerPS', 'kilometer', 'price']),
#                     # ('price_filter', 'passthrough', [0], lambda x: (x > 100) & (x < 60000)),
#                     # ('powerPS_filter', 'passthrough', [1], lambda x: (x > 100) & (x < 2500)),
#                     # ('kilometer_filter', 'passthrough', [6], lambda x: (x > 5000) & (x < 1500000)),
#                     # ('yearOfRegistration', 'passthrough', [5], lambda x: (x > 1940) & (x < 2020)),
#                     # ('remove_andere', 'drop', [7], lambda x: x != 'andere'),
#                     # ('days_used', DaysUsedTransformer(date_column='lastSeen', reg_column='yearOfRegistration'), [8, 9]),
#                     # ('scaler', StandardScaler(), [1, 6, 0]),
#                     # ('drop_columns', 'drop', [10, 11]),
#                     # ('target_encode', self.target_encoder, [7, 8, 9, 2, 3, 4]),
#                     # ('standardize', self.scaler, [0, 1, 2, 3, 4, 5, 6, 7])
#                     # ('move_column', 'passthrough', )

#             preprocessor = Pipeline(
#                 [
#                     ('preprocessing_one', preprocessing_one),
#                     ('feature_engineering', feature_engineering),
#                     ('preprocessing_two', preprocessing_two)
#                 ]
#             )
            
#             return preprocessor

#         except Exception as e:
#             raise CustomException(e, sys)

        
#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)

#             print('train_df before removing NULL values')
#             print(train_df.info())

#             # This thing will delete the rows which contain null values
#             train_df = train_df.dropna(axis=0)
#             test_df = test_df.dropna(axis=0)
#             logging.info("Deletion of rows containing NaN value is done.")

#             print("train_df After deletion of NULL values.")
#             print(train_df.info())


#             # df = pd.read_csv(os.path.join('artifacts','data.csv'))

#             # print(train_df.columns)

#             logging.info("Read train and test data completed")

#             logging.info("Obtaining preprocessing object")

#             # This line will remove the rows which contain the price less than 100 and greater than 60000 euros
#             train_df = train_df[(train_df['price']>100) & (train_df['price']<60000)]
#             test_df = test_df[(test_df['price']>100) & (test_df['price']<60000)]
#             logging.info('Deletion of those rows containing price greater than 60000 and less than 100 euros is done.')

#             print('train_df After deletion of those rows in which the price is greater than 60000 and less than 100 euros.')
#             print(train_df.info())


#             preprocessing_obj = self.get_data_transformer_object()

#             target_column_name = "price"

#             input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
#             target_feature_train_df=train_df[target_column_name]

#             input_feature_test_df=test_df
#             target_feature_test_df=test_df[target_column_name]

#             logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")


#             # print("This is inputfeaturetestdf.columns : ",input_feature_train_df.columns, input_feature_train_df)

#             # X,Y =(self.list2int(input_feature_train_df))

#             # print(X,Y)

#             # print(tuple(input_feature_train_df))
             
#             print(input_feature_train_df.info())

#             # input_feature=preprocessing_obj.fit_transform(((input_feature_train_df)))
#             # input_feature_train_arr = preprocessing_obj.transform(input_feature_train_arr)
#             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_df)
#             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
#             train_arr = np.c_[
#                 input_feature_train_arr, np.array(target_feature_train_df)
#             ]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info(f"Saved preprocessing object.")

#             save_object(

#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj

#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path
#             )

#         except Exception as e:
#             raise CustomException(e, sys)/

import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# # Custom transformer to calculate 'daysUsed'
# class DaysUsedTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, date_column, reg_column):
#         self.date_column = date_column
#         self.reg_column = reg_column
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         X_copy = X.copy()
#         X_copy['daysUsed'] = (pd.to_datetime(X_copy[self.date_column]).dt.year - X_copy[self.reg_column])
#         return X_copy.drop(columns=[self.date_column, self.reg_column])

# class RemoveZerosTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns):
#         self.columns = columns
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         return X[(X[self.columns] != 0).all(axis=1)]



class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


def yearOfRegistration_filter_transform(X):
    print(X)
    return (X[(X > 1940) & (X < 2016)])


def kilometer_filter_transform(X):
    mean_value = X[(X > 5000) & (X < 150000)].mean()
    X[X < 5000] = mean_value
    X[X > 150000] = mean_value
    return X



def powerPS_filter_transform(X):
    mean_value = X[(X > 100) & (X < 2500)].mean()
    X[X < 100] = mean_value
    X[X > 2500] = mean_value
    return X

def remove_andere_filter_transform(X): # used 3 times
    print(X)
    return (X[X != 'andere'])

def year_used_filter_transform(Y):
    # Convert to numeric and coerce errors to NaN
    # numeric_values = pd.to_numeric((Y)) 
    
    # Convert to integers, ignoring NaN values
    
    print("data type of Y : ",type(Y[0,0]), "Size of Y : ", np.size(Y))
    print(Y)
    # for i in range(np.size(Y)):
    Z = 2016 - np.array(Y)  # Calculate the difference
    return pd.DataFrame(Z, columns=['yearUsed'])


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_encoder = TargetEncoder()
        self.scaler = StandardScaler()

    # def list2int(list):
    #     return int(list[0]), int(list[1])
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            """
            yearUsed : 0
            abtest : 1
            vehicleType = 2
            gearbox = 3
            powerPS = 4
            model = 5
            kilometer = 6
            fuelType = 7
            brand = 8
            notRepairedDamage = 9
            """
            categorical_columns = [1, 2, 3, 5, 7, 8, 9]

            numerical_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            cat_pipeline=Pipeline(

                steps=[
                ("TargetEncoding",TargetEncoder(target_type="continuous",smooth="auto"))
                ]
            )

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )

            preprocessing_one = ColumnTransformer(
                [
                    ('drop_index', 'drop', [0]),
                    ('drop_dateCrawled', 'drop', [1]),
                    ('drop_name', 'drop', [2]),
                    ('drop_seller', 'drop', [3]),
                    ('offerType', 'drop', [4]),
                    # ('abtest', 'passthrough',[5]),
                    # ('vehicleType', 'passthrough', [6]),
                    # ('yearOfRegistration', 'passthrough',[7]),
                    # ('gearbox','passthrough',[8]),
                    ('drop_monthOfRegistration', 'drop', [12]),
                    # ('notRepairedDamage','passthrough',[15]),
                    ('drop_dateCreated', 'drop', [16]),
                    ('drop_nrOfPictures', 'drop', [17]),
                    ('drop_postalCode', 'drop', [18]),
                    ('drop_lastSeen', 'drop', [19])
                    # ('powerPS_filter', FunctionTransformer(powerPS_filter_transform), [9]),
                    # ('kilometer_filter', FunctionTransformer(kilometer_filter_transform), [11]),
                    # ('yearOfRegistration', FunctionTransformer(yearOfRegistration_filter_transform),[7]),
                    # ('remove_andere_from_brand', FunctionTransformer(remove_andere_filter_transform), [14]),
                    # ('remove_andere_from_model', FunctionTransformer(remove_andere_filter_transform), [10]),
                    # ('remove_andere_from_fuelType', FunctionTransformer(remove_andere_filter_transform), [13])
                ], remainder='passthrough'
            )


            feature_engineering = ColumnTransformer(
                [
                    ('year_used', FunctionTransformer(year_used_filter_transform), tuple([2]))
                ], remainder='passthrough'
            )


            preprocessing_two = ColumnTransformer(
                [
                    # ('drop_yearOfRegistration', FunctionTransformer(year_used_filter_transform), tuple([9])),  # Index of 'yearOfRegistration' after previous transformations
                    # ('target_encode', TargetEncoder(), tuple([3, 4, 5, 6, 7, 8, 9])),  # Indices of columns to target encode
                    # ('standardize', self.scaler, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Indices of columns to standardize

                    ("cat_pipelines",cat_pipeline,categorical_columns)
                       
                ],remainder='passthrough'
            )
            preprocessing_three = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)

                ],remainder='passthrough'
            )

            preprocessor = Pipeline(
                [
                    ('preprocessing_one', preprocessing_one),
                    ('feature_engineering', feature_engineering),
                    ('preprocessing_two', preprocessing_two),
                    ('preprocessing_three', preprocessing_three)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print('train_df before removing NULL values')
            print(train_df.info())

            # Drop rows with NaN values
            train_df = train_df.dropna(axis=0)
            test_df = test_df.dropna(axis=0)
            logging.info("Deletion of rows containing NaN value is done.")

            print("train_df After deletion of NULL values.")
            print(train_df.info())
            print(train_df.isnull().sum())

            # Remove rows with price outliers
            train_df = train_df[(train_df['price'] > 100) & (train_df['price'] < 60000)]
            test_df = test_df[(test_df['price'] > 100) & (test_df['price'] < 60000)]
            logging.info('Deletion of rows with price outliers is done.')

            print('train_df After deletion of price outliers.')
            print(train_df.info())

            # Remove rows with powerPS outliers
            train_df = train_df[(train_df['powerPS'] > 100) & (train_df['powerPS'] < 2500)]
            test_df = test_df[(test_df['powerPS'] > 100) & (test_df['powerPS'] < 2500)]
            logging.info('Deletion of rows with powerPS outliers is done.')
            
            print('train_df After deletion of powerPS outliers.')
            print(train_df.info())

            # Remove rows with kilometer outliers
            train_df = train_df[(train_df['kilometer'] > 25000) & (train_df['kilometer'] < 225000)]
            test_df = test_df[(test_df['kilometer'] > 25000) & (test_df['kilometer'] < 225000)]
            logging.info('Deletion of rows with kilometer outliers is done.')

            print('train_df After deletion of kilometer outliers.')
            print(train_df.info())

            # Remove rows with yearOfRegistration outliers
            train_df = train_df[(train_df['yearOfRegistration'] > 1940) & (train_df['yearOfRegistration'] < 2016)]
            test_df = test_df[(test_df['yearOfRegistration'] > 1940) & (test_df['yearOfRegistration'] < 2016)]
            logging.info('Deletion of rows with yearOfRegistration outliers is done.')

            print('train_df After deletion of yearOfRegistration outliers.')
            print(train_df.info())

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # printing the info of target_feature_train_df
            print("printing the info of target_feature_train_df")
            print(target_feature_train_df.info())

            input_feature_test_df = test_df
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            

            print(input_feature_train_df.info())

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)

            # This print the name of features
            logging.info("Preprocessing one is done.")
            # print(preprocessing_obj.get_feature_names_out())
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Object_transformation is done.')

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            # train_arr_df = pd.DataFrame(train_arr)
            # test_arr_df = pd.DataFrame(test_arr)
            # train_arr_df.to_csv('notebook/train_arr_csv.csv')
            # test_arr_df.to_csv('notebook/test_arr_csv.csv')


            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

