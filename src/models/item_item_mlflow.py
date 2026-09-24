
import json
import pickle

import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ItemItemModel(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for the Item-Item Collaborative Filtering model.

    This class allows the trained Item-Item model to be saved,
    loaded, and used through the MLflow Model Registry.

    The model expects three inputs:

    - user_idx: Internal index of the user.
    - seen_items: List of item indices the user has interacted with.
    - N: Number of recommendations required.

    The user's item history is converted into a sparse matrix
    because the implicit Item-Item recommender expects the user's
    interaction data in sparse matrix format.

    The function then generates personalized item recommendations
    and returns the recommended item IDs along with their scores.
    """

    def __init__(self, n_items: int):
        """
        Initialize the Item-Item MLflow model.

        Parameters
        ----------
        n_items : int
            Total number of items in the Item-Item model.
        """

        self.n_items = n_items
        self.item_model = None

    def load_context(self, context):
        """
        Load the trained Item-Item model when MLflow loads the model.

        MLflow provides the path of the logged model artifact
        through context.artifacts. The saved pickle file is then
        loaded into self.item_model.
        """

        model_path = context.artifacts["item_model"]

        with open(model_path, "rb") as f:
            self.item_model = pickle.load(f)

    def predict(self, context, model_input, params=None):
        """
        Generate personalized recommendations for users.

        Parameters
        ----------
        context : MLflow context
            MLflow model execution context.

        model_input : pandas.DataFrame
            Input DataFrame containing:
                - user_idx
                - seen_items
                - N

            Example:
                user_idx = 100
                seen_items = [10, 25, 50]
                N = 10

        params : dict, optional
            Optional parameters provided by MLflow.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing:
                - user_idx
                - recommendations
                - scores
        """

        recommendations = []

        for _, row in model_input.iterrows():

            # Get the user's internal index
            user_idx = int(row["user_idx"])

            # Number of recommendations required
            N = int(row["N"])

            # Get items the user has already interacted with
            seen_items = row["seen_items"]

            # Convert JSON string to Python list if necessary
            if isinstance(seen_items, str):
                seen_items = json.loads(seen_items)

            # Make sure all item indices are integers
            seen_items = [int(item) for item in seen_items]

            # Create a sparse user-item interaction matrix
            user_items = csr_matrix(
                (
                    np.ones(len(seen_items)),
                    (
                        np.zeros(len(seen_items), dtype=int),
                        np.array(seen_items)
                    )
                ),
                shape=(1, self.n_items)
            )

            # Generate recommendations
            item_ids, scores = self.item_model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=N,
                filter_already_liked_items=True
            )

            # Store recommendations for this user
            recommendations.append(
                {
                    "user_idx": user_idx,
                    "recommendations": item_ids.tolist(),
                    "scores": scores.tolist()
                }
            )

        # Return results after processing all users
        return pd.DataFrame(recommendations)
