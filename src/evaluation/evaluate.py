import numpy as np


K_VALUES = [5, 10, 20, 50, 100]


import numpy as np
import pandas as pd


def build_eval_set(
    test_df: pd.DataFrame,
    user_to_idx: dict,
    item_to_idx: dict,
    user_item_matrix
):
    """
    Converts test-set users/items into internal indices
    and determines whether the target item was already
    present in the user's training history.
    """

    eval_user_idx = []
    eval_targets = []
    eval_already_seen = []

    for _, row in test_df.iterrows():

        visitor_id = row["visitorid"]
        item_id = row["itemid"]

        # User must exist in training mappings
        if visitor_id not in user_to_idx:
            continue

        # Item must exist in training mappings
        if item_id not in item_to_idx:
            continue

        user_idx = user_to_idx[visitor_id]
        item_idx = item_to_idx[item_id]

        # Check whether this user already interacted
        # with this item in the training data.
        already_seen = (
            user_item_matrix[user_idx, item_idx] > 0
        )

        eval_user_idx.append(user_idx)
        eval_targets.append(item_id)
        eval_already_seen.append(already_seen)

    return (
        eval_user_idx,
        eval_targets,
        np.array(eval_already_seen)
    )



def evaluate(
    recommend_fn,
    eval_user_idx,
    eval_targets,
    idx_to_item,
    k_values=K_VALUES,
    user_subset=None
):

    max_k = max(k_values)

    hits = {
        k: 0
        for k in k_values
    }

    total = 0

    if user_subset is None:
        indices = range(len(eval_user_idx))
    else:
        indices = np.where(user_subset)[0]

    for i in indices:

        user_idx = eval_user_idx[i]
        actual_item = eval_targets[i]

        ranked_idx = recommend_fn(
            user_idx,
            max_k
        )

        ranked_items = [
            idx_to_item[int(j)]
            for j in ranked_idx
        ]

        for k in k_values:

            if actual_item in ranked_items[:k]:
                hits[k] += 1

        total += 1

    # --------------------------------------------------
    # Handle case where there are no evaluation users
    # --------------------------------------------------

    if total == 0:

        recall_metrics = {
            f"Recall@{k}": 0.0
            for k in k_values
        }

        return recall_metrics, total

    # --------------------------------------------------
    # Calculate Recall@K
    # --------------------------------------------------

    recall_metrics = {
        f"Recall@{k}": hits[k] / total
        for k in k_values
    }

    # --------------------------------------------------
    # Return metrics + number of evaluated users
    # --------------------------------------------------

    return recall_metrics, total