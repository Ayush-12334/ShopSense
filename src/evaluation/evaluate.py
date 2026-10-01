import numpy as np
import pandas as pd

K_VALUES = [5, 10, 20, 50, 100]


def build_eval_set(test_df: pd.DataFrame, user_to_idx: dict, item_to_idx: dict, user_item_matrix):
    """
    Converts test-set users/items into internal indices and determines
    whether the target item was already present in the user's training history.
    """
    eval_user_idx = []
    eval_targets = []
    eval_already_seen = []

    for _, row in test_df.iterrows():
        visitor_id = row["visitorid"]
        item_id = row["itemid"]

        if visitor_id not in user_to_idx:
            continue
        if item_id not in item_to_idx:
            continue

        user_idx = user_to_idx[visitor_id]
        item_idx = item_to_idx[item_id]
        already_seen = user_item_matrix[user_idx, item_idx] > 0

        eval_user_idx.append(user_idx)
        eval_targets.append(item_id)
        eval_already_seen.append(already_seen)

    return eval_user_idx, eval_targets, np.array(eval_already_seen, dtype=bool)


def build_category_mapping(catalog_df: pd.DataFrame):
    """Build model_item_idx -> category_id mapping from product catalog."""
    required_columns = {"model_item_idx", "category_id"}
    missing = required_columns - set(catalog_df.columns)

    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    category_mapping = {}

    for _, row in catalog_df.iterrows():
        item_idx = row["model_item_idx"]
        category_id = row["category_id"]

        if pd.isna(item_idx):
            continue
        if pd.isna(category_id):
            category_id = "unknown"

        category_mapping[int(item_idx)] = category_id

    return category_mapping


def build_item_popularity(train_df: pd.DataFrame, item_to_idx: dict):
    """Calculate item popularity from training interactions."""
    if "itemid" not in train_df.columns:
        raise ValueError("train_df must contain an 'itemid' column")

    item_counts = train_df["itemid"].value_counts()
    total_interactions = item_counts.sum()

    if total_interactions == 0:
        raise ValueError("Training data contains no item interactions.")

    item_popularity = {}

    for item_id, count in item_counts.items():
        if item_id not in item_to_idx:
            continue

        item_idx = item_to_idx[item_id]
        item_popularity[int(item_idx)] = float(count / total_interactions)

    return item_popularity


def calculate_diversity(ranked_items, k, category_mapping):
    """Diversity@K = unique categories / number of recommended items."""
    top_items = ranked_items[:k]

    if not top_items:
        return 0.0

    categories = [
        category_mapping.get(int(item_idx), "unknown")
        for item_idx in top_items
    ]

    return len(set(categories)) / len(top_items)


def calculate_novelty(ranked_items, k, item_popularity):
    """Novelty@K using -log2(item popularity)."""
    top_items = ranked_items[:k]

    if not top_items:
        return 0.0

    novelty_scores = []

    for item_idx in top_items:
        probability = item_popularity.get(int(item_idx))

        if probability is None or probability <= 0:
            continue

        novelty_scores.append(float(-np.log2(probability)))

    return float(np.mean(novelty_scores)) if novelty_scores else 0.0


def _calculate_group_metrics(group_indices, eval_user_idx, eval_targets, idx_to_item,
                             recommendations, category_mapping, item_popularity, k_values):
    """Calculate Recall/Coverage/Diversity/Novelty for one evaluation group."""
    group_indices = list(group_indices)
    total = len(group_indices)

    if total == 0:
        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"Coverage@{k}"] = 0.0
            metrics[f"Diversity@{k}"] = 0.0
            metrics[f"Novelty@{k}"] = 0.0
        return metrics, 0

    hits = {k: 0 for k in k_values}
    recommended_items = {k: set() for k in k_values}
    diversity_values = {k: [] for k in k_values}
    novelty_values = {k: [] for k in k_values}

    for i in group_indices:
        actual_item = eval_targets[i]
        ranked_idx = recommendations[i]
        ranked_items = [idx_to_item[int(j)] for j in ranked_idx]

        for k in k_values:
            top_k = ranked_items[:k]

            if actual_item in top_k:
                hits[k] += 1

            recommended_items[k].update(top_k)

            diversity_values[k].append(
                calculate_diversity(ranked_idx, k, category_mapping)
            )

            novelty_values[k].append(
                calculate_novelty(ranked_idx, k, item_popularity)
            )

    catalog_size = len(idx_to_item)
    metrics = {}

    for k in k_values:
        metrics[f"Recall@{k}"] = hits[k] / total
        metrics[f"Coverage@{k}"] = (
            len(recommended_items[k]) / catalog_size
            if catalog_size > 0 else 0.0
        )
        metrics[f"Diversity@{k}"] = (
            float(np.mean(diversity_values[k]))
            if diversity_values[k] else 0.0
        )
        metrics[f"Novelty@{k}"] = (
            float(np.mean(novelty_values[k]))
            if novelty_values[k] else 0.0
        )

    return metrics, total


def evaluate(recommend_fn, eval_user_idx, eval_targets, idx_to_item, catalog_df,
             train_df, item_to_idx, eval_already_seen, k_values=K_VALUES, user_subset=None):
    """
    Calculates Recall, Coverage, Diversity and Novelty overall,
    separately for repeat and novel targets.
    """
    max_k = max(k_values)

    category_mapping = build_category_mapping(catalog_df)
    item_popularity = build_item_popularity(train_df, item_to_idx)

    indices = (
        range(len(eval_user_idx))
        if user_subset is None
        else np.where(user_subset)[0]
    )

    recommendation_map = {
        i: recommend_fn(eval_user_idx[i], max_k)
        for i in indices
    }

    overall_indices = list(recommendation_map.keys())

    overall_metrics, overall_total = _calculate_group_metrics(
        overall_indices,
        eval_user_idx,
        eval_targets,
        idx_to_item,
        recommendation_map,
        category_mapping,
        item_popularity,
        k_values
    )

    repeat_indices = [
        i for i in overall_indices
        if eval_already_seen[i]
    ]

    repeat_metrics, repeat_total = _calculate_group_metrics(
        repeat_indices,
        eval_user_idx,
        eval_targets,
        idx_to_item,
        recommendation_map,
        category_mapping,
        item_popularity,
        k_values
    )

    novel_indices = [
        i for i in overall_indices
        if not eval_already_seen[i]
    ]

    novel_metrics, novel_total = _calculate_group_metrics(
        novel_indices,
        eval_user_idx,
        eval_targets,
        idx_to_item,
        recommendation_map,
        category_mapping,
        item_popularity,
        k_values
    )

    metrics = {}
    metrics.update(overall_metrics)
    metrics.update({
        f"Repeat_{key}": value
        for key, value in repeat_metrics.items()
    })
    metrics.update({
        f"Novel_{key}": value
        for key, value in novel_metrics.items()
    })

    metrics["Total_Evaluation_Users"] = overall_total
    metrics["Repeat_Target_Count"] = repeat_total
    metrics["Novel_Target_Count"] = novel_total

    return metrics, overall_total