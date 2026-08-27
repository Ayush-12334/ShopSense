import sys
import os
import numpy as np 
import pandas as pd 
from src.logger import logging
from src.exception import CustomeException
from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        Type hint: data_ingestion_config should be a DataIngestionConfig object.
        Default value: if no config is passed, create a new DataIngestionConfig object.
        
        """
        self.data_ingestion_config=data_ingestion_config



    def initiate_data_ingestion(self):
        """
        Method_name : initate_data_ingestion
        Description : This method save the 'RAW_CSV' file from scoure 
        """
        try:
            logging.info("initiating the process of data_ingestion")
            df=pd.read_csv("datasets/events.csv")
            feature_store_directory=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_directory,exist_ok=True)
            df.to_csv(self.data_ingestion_config.feature_store_file_path,index=False)
            logging.info("The file saved sucessfully")
            return df
   
        except Exception as e:

            raise CustomeException(e,sys) from e
    

    def initiate_train_test_split(self, dataframe: pd.DataFrame) -> DataIngestionArtifacts:

        """
        Method_name: initiate_train_test_split
        Description : This method split the data into train adn test after cheking all the condition mention or experimented 
        the NoteBook  
        
        """

        try:
            logging.info("started with train_test_split")

            events_df = dataframe.copy()  # use what was passed in, not a re-read from disk
            events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], unit="ms")

            transaction = events_df[events_df['event'] == 'transaction'].copy()

            test_df = (
                transaction.sort_values("timestamp")
                .groupby('visitorid')
                .tail(1)
                .reset_index(drop=True)
            )
            cutoff_by_user = test_df.set_index('visitorid')['timestamp']

            events_df['cutoff'] = events_df['visitorid'].map(cutoff_by_user)

            train = events_df['cutoff'].isna() | (events_df['timestamp'] < events_df['cutoff'])
            train_events = events_df[train].drop(columns='cutoff').copy()
            test_events = test_df.copy()

        # Leakage check -- fails loudly if this is ever broken again.
            check = train_events.merge(
                cutoff_by_user.rename('cutoff'), left_on='visitorid', right_index=True, how='inner'
            )
            assert (check['timestamp'] < check['cutoff']).all(), \
                "LEAKAGE: some training events occur at or after the held-out transaction timestamp."

            logging.info(f"Train events: {len(train_events):,} / {len(events_df):,} total_events")

            ingested_directory = self.data_ingestion_config.ingested_dir
            os.makedirs(ingested_directory, exist_ok=True)

            train_file = self.data_ingestion_config.training_file_path
            test_file = self.data_ingestion_config.testing_file_path

            train_events.to_csv(train_file, index=False)
            test_events.to_csv(test_file, index=False)

            logging.info("Train and test datasets saved successfully")

            return DataIngestionArtifacts(
                train_file_path=train_file,
                test_file_path=test_file,
            )

        except Exception as e:
            raise CustomeException(e, sys) from e

