from src.exception import CustomeException
from src.logger import logging
from src.constants import *
import sys
import numpy as np
import pandas as pd 
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.entity.config_entity import ModelTrainerConfig


class SASRecDataset(Dataset):
    def __init__(self, sequence, n_items, max_len=SASREC_MAX_SEQ_LEN):
        self.sequence = sequence
        self.n_items = n_items
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        seq = [i + 1 for i in self.sequence[idx]][-(self.max_len + 1):]
        input_seq, target_seq = seq[:-1], seq[1:]

        pad_len = self.max_len - len(input_seq)
        input_seq = [PAD] * pad_len + input_seq
        target_seq = [PAD] * pad_len + target_seq

        target_set = set(seq)
        neg_seq = []

        for target in target_seq:
            if target == PAD:
                neg_seq.append(PAD)
                continue

            neg = np.random.randint(1, self.n_items + 1)
            while neg in target_set:
                neg = np.random.randint(1, self.n_items + 1)

            neg_seq.append(neg)

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long)
        )


class SASRec(nn.Module):
    def __init__(
        self,
        n_items,
        max_len=SASREC_MAX_SEQ_LEN,
        d_model=SASREC_D_MODEL,
        n_heads=SASREC_N_HEADS,
        n_layers=SASREC_N_LAYERS,
        dropout=0.2
    ):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model

        self.item_emb = nn.Embedding(
            n_items + 1,
            d_model,
            padding_idx=PAD
        )

        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.layers_norm = nn.LayerNorm(d_model)

    def forward(self, input_seq):
        position = torch.arange(
            input_seq.size(1),
            device=input_seq.device
        ).unsqueeze(0)

        x = self.item_emb(input_seq) + self.pos_emb(position)
        x = self.dropout(x)

        seq_len = input_seq.size(1)

        causal_mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=input_seq.device
            ),
            diagonal=1
        )

        padding_mask = input_seq == PAD

        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask
            )

        return self.layers_norm(x)

    def score_items(self, hidden, item_ids):
        item_vecs = self.item_emb(item_ids)
        return (hidden * item_vecs).sum(-1)


class SASREC:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()
    ):
        self.config = model_trainer_config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

        logging.info(
            f"SASRec training device: {self.device}"
        )

    def create_sequence(self, train_events, item_to_idx):
        try:
            logging.info("Creating SASRec session sequences")
             # Copy original training data

            session_source = train_events.copy()

            if pd.api.types.is_numeric_dtype(session_source['timestamp']):
                session_source['timestamp']=pd.to_datetime(
                    session_source['timestamp'],
                    unit='ms',
                    errors='coerce'
                )
            else:
                session_source['timestamp']=pd.to_datetime(
                    session_source['timestamp'],
                    errors='coerce'
                            )





           
            session_source=session_source.dropna(
                subset=['timestamp']
            )

            session_source = (
                session_source
                .sort_values(['visitorid', 'timestamp'])
                .copy()
                
            )

            session_source['time_diff'] = (
                session_source
                .groupby('visitorid')['timestamp']
                .diff()
                .dt.total_seconds()
                .fillna(0)
                
            )

            session_source['new_session'] = (
                session_source['time_diff']
                > self.config.session_gap_minutes * 60
            ).astype(int)

            session_source['session_num'] = (
                session_source
                .groupby('visitorid')['new_session']
                .cumsum()
            )

            session_source['item_idx'] = (
                session_source['itemid'].map(item_to_idx)
            )

            session_source = session_source.dropna(
                subset=['item_idx']
            )

            session_source['item_idx'] = (
                session_source['item_idx'].astype(int)
            )

            session_sequence = (
                session_source
                .groupby(['visitorid', 'session_num'])['item_idx']
                .apply(list)
                .tolist()
            )
            print("Total sessions:", len(session_sequence))


            logging.info(
                    f"5. Sequences before filtering: {len(session_sequence):,}"
                )   

            session_sequence = [
                [
                    item for i, item in enumerate(seq)
                    if i == 0 or item != seq[i - 1]
                ]
                for seq in session_sequence
            ]

            logging.info(
                f"6. Sequences after removing repeats: {len(session_sequence):,}"
            )
            print("After removing repeats:", len(session_sequence))
            print("Example sessions:", session_sequence[:5])

            session_sequence = [
                seq for seq in session_sequence
                if len(seq) >= 2
            ]

            logging.info(
                f"7. FINAL sequences >= 2 items: {len(session_sequence):,}"
            )
            
            print("Sessions with >= 2 items:", len(session_sequence))

            session_sequence = [
                seq[-self.config.sasrec_max_seq_len:]
                for seq in session_sequence
            ]

            logging.info(
                f"Created {len(session_sequence):,} valid SASRec sequences"
            )

            return session_sequence

        except Exception as e:
            raise CustomeException(e, sys) from e

    def create_dataloader(self, session_sequences, n_items):
        try:
            dataset = SASRecDataset(
                sequence=session_sequences,
                n_items=n_items,
                max_len=self.config.sasrec_max_seq_len
            )

            loader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=True
            )

            logging.info(
                f"SASRec DataLoader created: {len(dataset):,} sequences"
            )

            return loader

        except Exception as e:
            raise CustomeException(e, sys) from e

    def train(self, session_sequences, n_items):
        try:
            logging.info("Starting SASRec training")

            loader = self.create_dataloader(
                session_sequences,
                n_items
            )

            self.model = SASRec(
                n_items=n_items,
                max_len=self.config.sasrec_max_seq_len,
                d_model=self.config.sasrec_d_model,
                n_heads=self.config.sasrec_n_heads,
                n_layers=self.config.sasrec_n_layers
            ).to(self.device)

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1e-3
            )

            bce = nn.BCEWithLogitsLoss(reduction='none')

            for epoch in range(self.config.sasrec_n_epochs):
                self.model.train()
                total_loss = 0.0

                for input_seq, target_seq, neg_seq in loader:
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    neg_seq = neg_seq.to(self.device)

                    hidden = self.model(input_seq)

                    pos_logits = self.model.score_items(
                        hidden, target_seq
                    )

                    neg_logits = self.model.score_items(
                        hidden, neg_seq
                    )

                    mask = (target_seq != PAD).float()

                    pos_loss = (
                        bce(
                            pos_logits,
                            torch.ones_like(pos_logits)
                        ) * mask
                    )

                    neg_loss = (
                        bce(
                            neg_logits,
                            torch.zeros_like(neg_logits)
                        ) * mask
                    )

                    loss = (
                        (pos_loss + neg_loss).sum()
                        / mask.sum().clamp_min(1.0)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)

                logging.info(
                    f"Epoch {epoch + 1}/{self.config.sasrec_n_epochs} "
                    f"loss={avg_loss:.4f}"
                )

            logging.info("SASRec training completed")

            return self.model

        except Exception as e:
            raise CustomeException(e, sys) from e
        

   

    def build_last_session_by_user(self,train_events,item_to_idx):

        try:
            logging.info(
                "Building last session for each user"
            )

            sessions_source = train_events.copy()

        # ---------------------------------------------------------
        # 1. Timestamp conversion
        # ---------------------------------------------------------
            if not pd.api.types.is_datetime64_any_dtype(
                sessions_source["timestamp"]
            ):
                sessions_source["timestamp"] = pd.to_datetime(
                    sessions_source["timestamp"],
                    unit="ms",
                    errors="coerce"
                )

            sessions_source = sessions_source.dropna(
                subset=["timestamp"]
            )

        # ---------------------------------------------------------
        # 2. Sort
        # ---------------------------------------------------------
            sessions_source = sessions_source.sort_values(
                ["visitorid", "timestamp"]
            ).copy()

        # ---------------------------------------------------------
        # 3. Time difference
        # ---------------------------------------------------------
            sessions_source["time_diff"] = (
                sessions_source
                .groupby("visitorid")["timestamp"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )

        # ---------------------------------------------------------
        # 4. Session boundary
        # ---------------------------------------------------------
            sessions_source["new_session"] = (
                sessions_source["time_diff"]
                > self.config.session_gap_minutes * 60
            ).astype(int)

        # ---------------------------------------------------------
        # 5. Session number
        # ---------------------------------------------------------
            sessions_source["session_num"] = (
                sessions_source
                .groupby("visitorid")["new_session"]
                .cumsum()
            )

        # ---------------------------------------------------------
        # 6. Item mapping
        # ---------------------------------------------------------
            sessions_source["item_idx"] = (
                sessions_source["itemid"].map(item_to_idx)
            )

            sessions_source = sessions_source.dropna(
                subset=["item_idx"]
            )

            sessions_source["item_idx"] = (
                sessions_source["item_idx"].astype(int)
            )

        # ---------------------------------------------------------
        # 7. Create sessions
        # ---------------------------------------------------------
            sessions = (
                sessions_source
                .groupby(
                    ["visitorid", "session_num"]
                )
                .agg(
                    items=("item_idx", list),
                    timestamp=("timestamp", "max")
                )
                .reset_index()
            )

        # ---------------------------------------------------------
        # 8. Remove consecutive repeats
        # ---------------------------------------------------------
            sessions["items"] = sessions["items"].apply(
                lambda seq: [
                    item
                    for i, item in enumerate(seq)
                    if i == 0 or item != seq[i - 1]
                ]
            )

        # ---------------------------------------------------------
        # 9. Keep valid sessions
        # ---------------------------------------------------------
            sessions = sessions[
                sessions["items"].apply(len) >= 2
            ]

        # ---------------------------------------------------------
        # 10. Last session per user
        # ---------------------------------------------------------
            last_sessions = (
                sessions
                .sort_values("timestamp")
                .groupby("visitorid")
                .tail(1)
            )

            last_session_by_user = dict(
                zip(
                    last_sessions["visitorid"],
                    last_sessions["items"]
                )
            )

            logging.info(
                f"Users with last session: "
                f"{len(last_session_by_user):,}"
            )

            return last_session_by_user

        except Exception as e:
         
            raise CustomeException(e, sys) from e

    def sasrec_score_candidates(
        self,
        visitor_id,
        candidate_item_idx,
        last_session_by_user,
        sasrec_model,
        device,
        max_seq_len=50  
    ):
        try:
            logging.info("creating score candidates")

                

            session = last_session_by_user.get(visitor_id)

            if not session:
                return None

        # Keep only last MAX_SEQ_LEN items
            seq = session[-max_seq_len:]

        # SASRec uses:
        # 0 = PAD
        # item indices = 1, 2, 3, ...
            seq = [item + 1 for item in seq]

        # Left padding
            pad_len = max_seq_len - len(seq)

            input_seq = torch.tensor(
                [[PAD] * pad_len + seq],
                dtype=torch.long,
                device=device
            )

            sasrec_model.eval()

            with torch.no_grad():

            # User/session representation
                hidden = sasrec_model(
                    input_seq
                )[0, -1]

            # Candidate items also need +1
                cand_tensor = torch.tensor(
                    [item + 1 for item in candidate_item_idx],
                    dtype=torch.long,
                    device=device
                )

            # Dot product:
            # user/session representation
            # × item embedding
                scores = (
                    hidden.unsqueeze(0)
                    * sasrec_model.item_emb(cand_tensor)
                ).sum(-1)

            return scores.cpu().numpy()
        except Exception as e :
            raise CustomeException(e,sys) from e