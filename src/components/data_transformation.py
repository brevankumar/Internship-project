import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler



#import category_encoders

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Categorical Features
            categorical_cols_for_OneHotEncoding = ['marital-status', 'relationship', 'race', 'sex']
            
            categorical_cols_for_OrdinalEncoding = ['workclass', 'education']

            categorical_cols_for_LabelEncoding = ['salary']

            
            # Define the custom ranking for each ordinal variable

            workclass_categories = [' Never-worked', ' Without-pay',' Self-emp-inc',' Self-emp-not-inc',' Local-gov',' State-gov',' Federal-gov',' Private']
            

            education_categories = [' Preschool',' 1st-4th',' 5th-6th', ' 7th-8th', ' 9th',' 10th',' 11th',' 12th',' Some-college',' HS-grad',
                                    ' Assoc-voc','Assoc-acdm', ' Bachelors',' Prof-school',' Masters',' Doctorate']


            # categories for one hot encoding
            
            marital_status_categories = [' Never-married', ' Married-civ-spouse', ' Divorced',' Married-spouse-absent', ' Separated', ' Married-AF-spouse', ' Widowed']

            relationship_categories = [' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',' Other-relative']

            race_categories = [' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other']
            
            sex_categories = [' Male', ' Female']



            logging.info('Pipeline Initiated')


            # Categorigal Pipeline
            categorical_pipeline_for_Onehotencoding = Pipeline(
                steps=[
                ('OneHotEncoder', OneHotEncoder(categories= [marital_status_categories,relationship_categories,race_categories,sex_categories], drop="first"))
                ]
                )
            
            categorical_pipeline_for_OrdinalEncoding = Pipeline(
                steps=[
                ('OrdinalEncoder', OrdinalEncoder(categories=[workclass_categories,education_categories]))
                ]
                )
            
            categorical_pipeline_for_LabelEncoding = Pipeline(
                steps=[
                ('LabelEncoder', LabelEncoder())
                ]
                )
                        


            preprocessor=ColumnTransformer(transformers= [('cat_pipeline_for_Onehotencoding',categorical_pipeline_for_Onehotencoding,categorical_cols_for_OneHotEncoding),
                                                    ('cat_pipeline_for_OrdinalEncoding',categorical_pipeline_for_OrdinalEncoding,categorical_cols_for_OrdinalEncoding),
                                                    ('cat_pipeline_for_LabelEncoding',categorical_pipeline_for_LabelEncoding,categorical_cols_for_LabelEncoding)],
    
                                                          remainder='passthrough',sparse_threshold=0)
                                                          
                                                          
                                                         

            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        

        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            def replace_question_mark_with_mode_in_train_data(train_path):
                try:
                    train_df_1 = pd.read_csv(train_path)

                    train_df_1["workclass"].replace(' ?', ' Private', inplace = True) # replacing ' ?' with most frequent value which is mode with Private
                    train_df_1["occupation"].replace(' ?', ' Prof-specialty', inplace = True) # replacing ' ?' with most frequent value which is mode with Prof-specialty
                    train_df_1["country"].replace(' ?', ' United-States', inplace = True) # replacing ' ?' with most frequent value which is mode with United-States
        
                    return train_df_1
        
    
                except Exception as e:
                   logging.info("Exception occured in the replacing with mode in train data") 
                    

                    
            def replace_question_mark_with_mode_in_test_data(test_path):
                try:
                    test_df_2 = pd.read_csv(test_path)
        
                    test_df_2["workclass"].replace(' ?', ' Private', inplace = True) # replacing ' ?' with most frequent value which is mode with Private
                    test_df_2["occupation"].replace(' ?', ' Prof-specialty', inplace = True) # replacing ' ?' with most frequent value which is mode with Prof-specialty
                    test_df_2["country"].replace(' ?', ' United-States', inplace = True) # replacing ' ?' with most frequent value which is mode with United-States                    
        

                    return test_df_2
        
    
                except Exception as e:
                   logging.info("Exception occured in the replacing with mode in test data")


            def frequency_encoding_in_train_data(train_path):
                try:
                    train_df_3 = pd.read_csv(train_path)
                   
                    df_frequency_map_1  = train_df_3.occupation.value_counts().to_dict()
                    train_df_3.occupation = train_df_3.occupation.map(df_frequency_map_1)

                    df_frequency_map_2 = train_df_3.country.value_counts().to_dict()
                    train_df_3.country = train_df_3.country.map(df_frequency_map_2)

                   
                    return train_df_3
        
    
                except Exception as e:               
                    logging.info("Exception occured for frequency encoding in train data")       



            def frequency_encoding_in_test_data(test_path):
                try:
                    test_df_4 = pd.read_csv(test_path)
                   
                    df_frequency_map_1  = test_df_4.occupation.value_counts().to_dict()
                    test_df_4.occupation = test_df_4.occupation.map(df_frequency_map_1)

                    df_frequency_map_2 = test_df_4.country.value_counts().to_dict()
                    test_df_4.country = test_df_4.country.map(df_frequency_map_2)
                   
                    return test_df_4
        
    
                except Exception as e:               
                    logging.info("Exception occured for frequency encoding in test data")

            
            def perform_label_encoding_in_traindata(train_path):
                try:
                    train_df_7 = pd.read_csv(train_path)
                  # Create a LabelEncoder instance
                    label_encoder = LabelEncoder()
        
                  # Fit and transform the data column
                    train_df_7["salary"] = label_encoder.fit_transform(train_df_7["salary"])

                  # Return the encoded column and the label encoder object
                
                    return  train_df_7
            
                except Exception as e:
                    logging.info("Exception occured for label encoding in train data")


            def perform_label_encoding_in_testdata(test_path):
                try:
                    test_df_8 = pd.read_csv(test_path)
                  # Create a LabelEncoder instance
                    label_encoder = LabelEncoder()
        
                  # Fit and transform the data column
                    test_df_8["salary"] = label_encoder.fit_transform(test_df_8["salary"])

                  # Return the encoded column and the label encoder object
                
                    return test_df_8
            
                except Exception as e:
                    logging.info("Exception occured for label encoding in test data")




            # Reading train and test data
            train_df_final=replace_question_mark_with_mode_in_train_data(train_path)
            test_df_final=replace_question_mark_with_mode_in_test_data(test_path)

            train_df_final_1 = frequency_encoding_in_train_data(train_df_final)
            test_df_final_1 = frequency_encoding_in_test_data(test_df_final)

            train_df_final_2 = perform_label_encoding_in_traindata(train_df_final_1)
            test_df_final_2 = perform_label_encoding_in_testdata(test_df_final_1)


            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df_final_2.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df_final_2.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            ## Transformating using preprocessor obj

            input_feature_train_arr = preprocessing_obj.fit_transform(train_df_final_2)
            input_feature_test_arr = preprocessing_obj.transform(test_df_final_2)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = input_feature_train_arr
            test_arr = input_feature_test_arr

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
                















               
""" def perform_random_oversampling_in_train_data(train_path):
                try:
                    train_df_5 = pd.read_csv(train_path)
                    X = train_df_5.drop(["salary"],axis=1)
                    y = train_df_5.salary
                    
                    rs = RandomOverSampler(random_state=30)
                    rs.fit(X,y)
                    X_train_resampled, y_train_resampled = rs.fit_resample(X, y)
        
                    return X_train_resampled,y_train_resampled
        
                except Exception as e:
                    logging.info("Exception occured for Random_Over_Sampler in train data")  


            def perform_random_oversampling_in_test_data(test_path):
                try:
                    test_df_6= pd.read_csv(test_path)
                    X = test_df_6.drop(["salary"],axis=1)
                    y = test_df_6.salary
                    
                    rs = RandomOverSampler(random_state=30)
                    rs.fit(X,y)
                    X_test_resampled, y_test_resampled = rs.fit_resample(X, y)
        
                    return X_test_resampled,y_test_resampled
        
                except Exception as e:
                    logging.info("Exception occured for Random_Over_Sampler in test data")"""
