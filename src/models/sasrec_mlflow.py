import torch
import mlflow.pyfunc
import numpy as np

from src.components.sasrec import SASRec


class SASRecModel(mlflow.pyfunc.PythonModel):

    def __init__(
        self,
        n_items,
        max_len,
        d_model,
        n_heads,
        n_layers,
        dropout,
        item_to_idx,
        idx_to_item,
        last_session_by_user
    ):
        self.n_items = n_items
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.last_session_by_user = last_session_by_user

    def load_context(self, context):

        self.device = torch.device("cpu")

        self.model = SASRec(
            n_items=self.n_items,
            max_len=self.max_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

        self.model.load_state_dict(
            torch.load(
                context.artifacts["sasrec_model"],
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input):

        recommendations = []

        for _, row in model_input.iterrows():

            visitor_id = row["visitor_id"]

            candidate_items = row["candidate_items"]

            session = self.last_session_by_user.get(visitor_id)

            if not session:
                recommendations.append([])
                continue

            # Keep only last max_len items
            session = session[-self.max_len:]

            # SASRec uses 0 as PAD
            # Original item indices are 0-based
            # SASRec item IDs are 1-based
            seq = [item + 1 for item in session]

            # Left padding
            pad_len = self.max_len - len(seq)

            input_seq = [0] * pad_len + seq

            input_tensor = torch.tensor(
                [input_seq],
                dtype=torch.long,
                device=self.device
            )

            with torch.no_grad():

                hidden = self.model(
                    input_tensor
                )[0, -1]

                candidate_indices = [
                    self.item_to_idx[item]
                    for item in candidate_items
                    if item in self.item_to_idx
                ]

                if not candidate_indices:
                    recommendations.append([])
                    continue

                candidate_tensor = torch.tensor(
                    [idx + 1 for idx in candidate_indices],
                    dtype=torch.long,
                    device=self.device
                )

                item_embeddings = self.model.item_emb(
                    candidate_tensor
                )

                scores = (
                    hidden.unsqueeze(0) * item_embeddings
                ).sum(-1)

                scores = scores.cpu().numpy()

            ranked_indices = np.argsort(
                -scores
            )

            ranked_items = [
                self.idx_to_item[candidate_indices[i]]
                for i in ranked_indices
            ]

            recommendations.append(ranked_items)

        return recommendations