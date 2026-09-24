from implicit.nearest_neighbours import ItemItemRecommender


def train_item_item(user_item_matrix, k=50):

    model = ItemItemRecommender(
        K=k
    )

    model.fit(
        user_item_matrix
    )

    return model