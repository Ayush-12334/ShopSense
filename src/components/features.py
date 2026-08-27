import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomeException
from scipy.sparse import csr_matrix
from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import FeatureArtifacts
from src.constants import EVENTS_WEIGHTS

class FeatureEngineering:
    def __init__(self,feature_engineering_config:FeatureEngineeringConfig=FeatureEngineeringConfig()):
        """
        Receive the feature engineering Configuration
        """

        self.feature_engineering_config=feature_engineering_config


    def build_interactions(self,train_events:pd.DataFrame) ->pd.DataFrame:
        try:
            logging.info("assigning the weights to to train set")
            # assign a weight based on the type of user event
            train_events=train_events.copy()
            train_events['weight']=train_events['event'].map(EVENTS_WEIGHTS)

            interaction_df=(train_events.groupby(['visitorid','itemid'],as_index=False
                                                 )
                                                 ['weight']
                                                 .sum()
                                                 )
            logging.info(f"Created{len(interaction_df):,} unique user-item interactions")
            
            return interaction_df

        except Exception as e:
            raise CustomeException(e,sys) from e
        

    def build_mappings(self,interaction_df:pd.DataFrame):
      


        try:
            logging.info("creating user and item mappings")

            # Get unique user and items
            user_ids=interaction_df['visitorid'].unique()
            item_ids=interaction_df['itemid'].unique()

            # convert original item ids to integer indices
            user_to_idx={
                user_id :idx for idx,user_id in enumerate(user_ids)
            }

            item_to_idx={
                item_id :idx for idx,item_id in enumerate(item_ids)
            }

            idx_to_user={
                idx:user_id for user_id,idx in user_to_idx.items()
            }

            idx_to_item={
                idx:item_id for item_id,idx in item_to_idx.items()
            }


            interaction_df=interaction_df.copy()


            interaction_df['user_idx']=(
                interaction_df['visitorid'].map(user_to_idx)
            )

            interaction_df['item_idx']=(
                interaction_df['itemid'].map(item_to_idx)
            )


            logging.info(
                f"Created mappings for"
                f"{len(user_to_idx):,} users and"
                f"{len(item_to_idx):,} items"
                )
            return (
                interaction_df,
                user_to_idx,
                item_to_idx,
                idx_to_user,
                idx_to_item
            )

        except Exception as e:
            raise CustomeException(e,sys) from e

    def build_sparse_matrices(self,interaction_df:pd.DataFrame,user_to_idx:dict,item_to_idx:dict):

        try:
            logging.info("creating user-item sparse matrices")

            n_users=len(user_to_idx)
            n_items=len(item_to_idx)

            user_item_matrix=csr_matrix(
                (
                    interaction_df['weight'].astype(np.float64),
                    (interaction_df['user_idx'],interaction_df['item_idx']),


               ),
               shape=(n_users,n_items)


            )
            item_user_matrix=user_item_matrix.T.tocsr()

            logging.info(f"Created user-item matrix with shape"f"{user_item_matrix.shape}")
            logging.info(f"Created item-user matrix with shape"f"{item_user_matrix.shape}")

            return user_item_matrix,item_user_matrix
        
        except Exception as e:
            raise CustomeException(e,sys) from e




    def initiate_feature_engineering(self,train_events:pd.DataFrame)->FeatureArtifacts:
        try:
            logging.info("Starting feature engineering")

            interaction_df=self.build_interactions(train_events=train_events)
            (
                interaction_df,
                user_to_idx,
                item_to_idx,
                idx_to_user,
                idx_to_item

            )=self.build_mappings(interaction_df=interaction_df)


            (
                user_item_matrix,
                item_user_matrix

            )=self.build_sparse_matrices(interaction_df=interaction_df,user_to_idx=user_to_idx,item_to_idx=item_to_idx)

            logging.info("feature engineering completed successfully")

            return FeatureArtifacts(

                interaction_df=interaction_df,
                user_to_idx=user_to_idx,
                item_to_idx=item_to_idx,
                idx_to_user=idx_to_user,
                idx_to_item=idx_to_item,
                user_item_matrix=user_item_matrix,
                item_user_matrix=item_user_matrix
                )



        except Exception as e:

            raise CustomeException(e,sys) from e
        
        

    


