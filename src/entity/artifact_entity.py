from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any
from scipy.sparse import csr_matrix
from typing import Optional

@dataclass
class DataIngestionArtifacts:
      train_file_path: str
      test_file_path: str


@dataclass
class FeatureArtifacts:
      """
      Contains all outputs generated during engineering
      """
      interaction_df:pd.DataFrame

      user_to_idx:Dict[Any,int]
      item_to_idx:Dict[Any,int]

      idx_to_user:Dict[int,Any]
      idx_to_item:Dict[int,Any]

      user_item_matrix:csr_matrix
      item_user_matrix:csr_matrix


@dataclass
class ModelTrainerArtifacts:
    item_item_model_path: str
    item_item_model_url: Optional[str]
    item_item_run_id: str
    item_item_metrics: dict

    sasrec_model_path: str
    sasrec_model_url: Optional[str]
    sasrec_run_id: Optional[str]
    sasrec_metrics: dict
    popularity_model_path: str
    popularity_model_url: Optional[str]
    popularity_run_id: str


