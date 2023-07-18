import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,PowerTransformer,MinMaxScaler,OneHotEncoder,TargetEncoder
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

            # Categorical & Numerical Features
            categorical_cols_1 = ['Item_Fat_Content', 'Outlet_Size','Outlet_Location_Type']
            
            categorical_cols_2 = ['Item_Type','Outlet_Identifier','Outlet_Type']
            
            categorical_cols_3 = ['Outlet_Establishment_Year']
            


            numerical_cols = ['Item_Weight', 'Item_Visibility','Item_MRP']

            
            # Define the custom ranking for each ordinal variable
            Item_Fat_Content_categories = ['Low Fat', 'Regular']
            Outlet_Size_categories = ['Small', 'Medium', 'High']
            Outlet_Location_Type_categories = ['Tier 3','Tier 2','Tier 1']
            
            # categories for one hot encoding
            Item_Type_categories = ['Dairy','Soft Drinks','Meat','Fruits and Vegetables','Household','Baking Goods','Snack Foods','Frozen Foods',
                                  'Breakfast','Health and Hygiene','Hard Drinks','Canned','Breads','Starchy Foods','Others','Seafood']

            Outlet_Identifier_categories = ['OUT049','OUT018','OUT010','OUT013','OUT027','OUT045','OUT017','OUT046','OUT035','OUT019']

            Outlet_Type_categories = ['Supermarket Type1','Supermarket Type2','Grocery Store','Supermarket Type3']

            # categories for target guided encoding
            Outlet_Establishment_Year_categories = ['1999', '2009', '1998', '1987', '1985', '2002', '2007', '1997', '2004']

            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',PowerTransformer())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline_1 = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories= [Item_Fat_Content_categories,Outlet_Size_categories,Outlet_Location_Type_categories])),
                ('scaler1', MinMaxScaler())
                ]
                )


            cat_pipeline_2 = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(categories = [Item_Type_categories,Outlet_Identifier_categories,Outlet_Type_categories] ,drop="first"))
                ]
                )
            
    
            """cat_pipeline_3 = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder2', OneHotEncoder(categories = Outlet_Establishment_Year_categories ,drop="first"))                
                ]
                )"""
            
            
            
            preprocessor=ColumnTransformer(transformers= [('num_pipeline',num_pipeline,numerical_cols), ('cat_pipeline1',cat_pipeline_1,categorical_cols_1),
                                                          ('cat_pipeline2',cat_pipeline_2,categorical_cols_2)],
                                                          remainder='passthrough',sparse_threshold=0)

            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            def replace_categories_traindata(train_path):
                try:
                    train_df_1 = pd.read_csv(train_path)
        
                    train_df_1["Item_Fat_Content"] = train_df_1["Item_Fat_Content"].map(
                            {"Low Fat" : 'Low Fat' , "LF" : "Low Fat" , 'low fat' : "Low Fat" , "Regular" : "Regular" , "reg" : "Regular" })
        
                    
                    return train_df_1
        
    
                except Exception as e:
                   logging.info("Exception occured in the replacing categories in train data") 
                    

                    
            def replace_categories_testdata(test_path):
                try:
                    test_df_2 = pd.read_csv(test_path)
        
                    test_df_2["Item_Fat_Content"] = test_df_2["Item_Fat_Content"].map(
                            {"Low Fat" : 'Low Fat' , "LF" : "Low Fat" , 'low fat' : "Low Fat" , "Regular" : "Regular" , "reg" : "Regular" })
                    
        

                    return test_df_2
        
    
                except Exception as e:
                   logging.info("Exception occured in the replacing categories in test data") 

            
            # Reading train and test data
            train_df = replace_categories_traindata(train_path)
            test_df =  replace_categories_testdata(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Item_Outlet_Sales'
            drop_columns = [target_column_name,'Item_Identifier']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Transformating using preprocessor obj

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

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
        
               
