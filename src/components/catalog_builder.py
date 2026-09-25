
import os
import pickle
import sys

import pandas as pd

from src.entity.config_entity import CatalogConfig
from src.logger import logging
from src.exception import CustomeException


class CatalogBuilder:

    def __init__(
        self,
        catalog_config: CatalogConfig = CatalogConfig()
    ):
        self.config = catalog_config

    # ==========================================================
    # GENERATE SYNTHETIC DISPLAY NAME
    # ==========================================================

    def _generate_display_name(
        self,
        item_id,
        category_id,
        category_path
    ):
        """
        Generate a deterministic synthetic product name.

        RetailRocket does not provide real product titles,
        so this name is only for demonstration purposes.

        The name is generated from the real RetailRocket
        item ID and category metadata.
        """

        try:

            if category_id is not None:

                if category_path:

                    category_label = "-".join(
                        str(category)
                        for category in category_path
                    )

                    return (
                        f"Product {item_id} "
                        f"- Category {category_label}"
                    )

                return (
                    f"Product {item_id} "
                    f"- Category {category_id}"
                )

            return (
                f"Product {item_id} "
                f"- Uncategorized"
            )

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # LOAD ITEM PROPERTIES
    # ==========================================================

    def _load_item_properties(self):

        try:

            logging.info(
                "Loading RetailRocket item properties"
            )

            frames = [
                pd.read_csv(path)
                for path in self.config.item_properties_paths
            ]

            properties = pd.concat(
                frames,
                ignore_index=True
            )

            required_columns = {
                "timestamp",
                "itemid",
                "property",
                "value"
            }

            missing = (
                required_columns
                - set(properties.columns)
            )

            if missing:

                raise ValueError(
                    f"Missing columns in item properties: {missing}"
                )

            properties["timestamp"] = pd.to_numeric(
                properties["timestamp"],
                errors="coerce"
            )

            properties = properties.dropna(
                subset=[
                    "timestamp",
                    "itemid"
                ]
            )

            properties["timestamp"] = (
                properties["timestamp"]
                .astype("int64")
            )

            logging.info(
                f"Item properties loaded: "
                f"{len(properties):,} rows"
            )

            return properties

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # LOAD CATEGORY TREE
    # ==========================================================

    def _load_category_tree(self):

        try:

            logging.info(
                "Loading category tree"
            )

            tree = pd.read_csv(
                self.config.category_tree_path
            )

            required_columns = {
                "categoryid",
                "parentid"
            }

            missing = (
                required_columns
                - set(tree.columns)
            )

            if missing:

                raise ValueError(
                    f"Missing columns in category tree: {missing}"
                )

            return tree

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # LOAD MODEL VOCABULARY
    # ==========================================================

    def _load_item_to_idx(self):

        try:

            logging.info(
                "Loading trained item vocabulary"
            )

            with open(
                self.config.item_to_idx_path,
                "rb"
            ) as file:

                item_to_idx = pickle.load(file)

            if not isinstance(
                item_to_idx,
                dict
            ):

                raise ValueError(
                    "item_to_idx.pkl does not contain a dictionary"
                )

            logging.info(
                f"Trained items found: "
                f"{len(item_to_idx):,}"
            )

            return item_to_idx

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # LOAD ACTUAL TRAIN DATA
    # ==========================================================

    def _load_train_events(self):

        try:

            logging.info(
                "Loading ShopSense training split"
            )

            train_path = (
                self.config.train_events_path
            )

            if not os.path.exists(train_path):

                raise FileNotFoundError(
                    f"Training file not found: {train_path}"
                )

            train_events = pd.read_csv(
                train_path
            )

            required_columns = {
                "timestamp",
                "itemid"
            }

            missing = (
                required_columns
                - set(train_events.columns)
            )

            if missing:

                raise ValueError(
                    f"Training file missing columns: {missing}"
                )

            train_events["timestamp"] = (
                pd.to_datetime(
                    train_events["timestamp"],
                    errors="coerce"
                )
            )

            train_events = train_events.dropna(
                subset=[
                    "timestamp",
                    "itemid"
                ]
            )

            logging.info(
                f"Training events loaded: "
                f"{len(train_events):,}"
            )

            return train_events

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # COMPUTE TRAINING CUTOFF
    # ==========================================================

    def _compute_cutoff(
        self,
        train_events
    ):

        try:

            cutoff_datetime = (
                train_events["timestamp"].max()
            )

            if pd.isna(cutoff_datetime):

                raise ValueError(
                    "Could not determine training cutoff"
                )

            # RetailRocket item_properties uses
            # Unix timestamp in milliseconds.

            cutoff = int(
                cutoff_datetime.timestamp() * 1000
            )

            logging.info(
                f"Catalog cutoff: "
                f"{cutoff_datetime}"
            )

            logging.info(
                f"Catalog cutoff timestamp(ms): "
                f"{cutoff}"
            )

            return cutoff

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # GET LATEST PROPERTY BEFORE CUTOFF
    # ==========================================================

    def _latest_property_at_cutoff(
        self,
        properties,
        property_name,
        cutoff
    ):

        try:

            subset = properties[
                (properties["property"] == property_name)
                &
                (properties["timestamp"] <= cutoff)
            ].copy()

            if subset.empty:

                return {}

            subset = subset.sort_values(
                "timestamp"
            )

            latest = (
                subset
                .groupby(
                    "itemid",
                    sort=False
                )
                .tail(1)
                .set_index("itemid")["value"]
                .to_dict()
            )

            return latest

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # BUILD CATEGORY HIERARCHY
    # ==========================================================

    def _build_category_paths(
        self,
        category_tree
    ):

        try:

            parent_of = (
                category_tree
                .set_index("categoryid")["parentid"]
                .to_dict()
            )

            paths = {}

            for category_id in parent_of:

                if pd.isna(category_id):

                    continue

                category_id = int(
                    category_id
                )

                path = [
                    category_id
                ]

                current = category_id

                for _ in range(
                    self.config.max_category_depth
                ):

                    parent = parent_of.get(
                        current
                    )

                    if parent is None:

                        break

                    if pd.isna(parent):

                        break

                    parent = int(
                        parent
                    )

                    if parent in path:

                        break

                    path.append(
                        parent
                    )

                    current = parent

                paths[category_id] = list(
                    reversed(path)
                )

            return paths

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # BUILD CATALOG
    # ==========================================================

    def build_catalog(self):

        try:

            logging.info(
                "========== PRODUCT CATALOG BUILD STARTED =========="
            )

            properties = (
                self._load_item_properties()
            )

            category_tree = (
                self._load_category_tree()
            )

            item_to_idx = (
                self._load_item_to_idx()
            )

            train_events = (
                self._load_train_events()
            )

            cutoff = (
                self._compute_cutoff(
                    train_events
                )
            )

            # --------------------------------------------------
            # Latest category for every item before cutoff
            # --------------------------------------------------

            category_by_item = (
                self._latest_property_at_cutoff(
                    properties,
                    "categoryid",
                    cutoff
                )
            )

            # --------------------------------------------------
            # Latest availability for every item before cutoff
            # --------------------------------------------------

            available_by_item = (
                self._latest_property_at_cutoff(
                    properties,
                    "available",
                    cutoff
                )
            )

            # --------------------------------------------------
            # Build category hierarchy
            # --------------------------------------------------

            category_paths = (
                self._build_category_paths(
                    category_tree
                )
            )

            # --------------------------------------------------
            # Count training interactions per item
            # --------------------------------------------------

            interaction_counts = (
                train_events
                .groupby("itemid")
                .size()
                .to_dict()
            )

            rows = []

            # ==================================================
            # BUILD ONE CATALOG ROW FOR EVERY TRAINED ITEM
            # ==================================================

            for item_id, model_idx in item_to_idx.items():

                # --------------------------------------------------
                # CATEGORY
                # --------------------------------------------------

                raw_category = (
                    category_by_item.get(
                        item_id
                    )
                )

                category_id = None

                if raw_category is not None:

                    try:

                        category_id = int(
                            float(raw_category)
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        category_id = None

                # --------------------------------------------------
                # AVAILABILITY
                # --------------------------------------------------

                raw_available = (
                    available_by_item.get(
                        item_id
                    )
                )

                available = None

                if raw_available is not None:

                    try:

                        available = int(
                            float(raw_available)
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        available = None

                # --------------------------------------------------
                # CATEGORY PATH
                # --------------------------------------------------

                category_path = None

                if category_id is not None:

                    category_path = (
                        category_paths.get(
                            category_id
                        )
                    )

                # --------------------------------------------------
                # SYNTHETIC DISPLAY NAME
                # --------------------------------------------------

                display_name = (
                    self._generate_display_name(
                        item_id=item_id,
                        category_id=category_id,
                        category_path=category_path
                    )
                )

                # --------------------------------------------------
                # FINAL CATALOG ROW
                # --------------------------------------------------

                rows.append({

                    # ==================================================
                    # REAL RETAILROCKET / MODEL INFORMATION
                    # ==================================================

                    "retailrocket_item_id": item_id,

                    "model_item_idx": int(
                        model_idx
                    ),

                    "category_id": category_id,

                    "category_path": category_path,

                    "available": available,

                    "train_interaction_count": int(
                        interaction_counts.get(
                            item_id,
                            0
                        )
                    ),

                    "has_category_metadata": (
                        category_id is not None
                    ),

                    # ==================================================
                    # SYNTHETIC USER-FACING INFORMATION
                    # ==================================================

                    "display_name": display_name,

                    "is_synthetic": True,

                })

            catalog = pd.DataFrame(
                rows
            )

            logging.info(
                f"Catalog rows created: "
                f"{len(catalog):,}"
            )

            logging.info(
                "========== PRODUCT CATALOG BUILD COMPLETED =========="
            )

            return catalog, item_to_idx

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def validate_catalog(
        self,
        catalog,
        item_to_idx
    ):

        try:

            trained_ids = set(
                item_to_idx.keys()
            )

            catalog_ids = set(
                catalog[
                    "retailrocket_item_id"
                ]
            )

            # --------------------------------------------------
            # Item vocabulary validation
            # --------------------------------------------------

            missing_items = (
                trained_ids
                - catalog_ids
            )

            extra_items = (
                catalog_ids
                - trained_ids
            )

            # --------------------------------------------------
            # Duplicate validation
            # --------------------------------------------------

            duplicate_count = int(
                catalog[
                    "retailrocket_item_id"
                ]
                .duplicated()
                .sum()
            )

            # --------------------------------------------------
            # Null item ID validation
            # --------------------------------------------------

            null_item_ids = int(
                catalog[
                    "retailrocket_item_id"
                ]
                .isna()
                .sum()
            )

            # --------------------------------------------------
            # Category metadata count
            # --------------------------------------------------

            category_count = int(
                catalog[
                    "has_category_metadata"
                ].sum()
            )

            # --------------------------------------------------
            # Display name validation
            # --------------------------------------------------

            null_display_names = int(
                catalog[
                    "display_name"
                ]
                .isna()
                .sum()
            )

            # --------------------------------------------------
            # Synthetic name count
            # --------------------------------------------------

            synthetic_name_count = int(
                catalog[
                    "is_synthetic"
                ].sum()
            )

            # --------------------------------------------------
            # Validation report
            # --------------------------------------------------

            report = {

                "trained_item_count":
                    len(trained_ids),

                "catalog_row_count":
                    len(catalog),

                "trained_items_missing":
                    len(missing_items),

                "catalog_items_not_in_vocab":
                    len(extra_items),

                "duplicate_item_ids":
                    duplicate_count,

                "null_item_ids":
                    null_item_ids,

                "items_with_category_metadata":
                    category_count,

                "null_display_names":
                    null_display_names,

                "synthetic_display_names":
                    synthetic_name_count,

            }

            print(
                "\n========================================"
            )

            print(
                "CATALOG VALIDATION"
            )

            print(
                "========================================"
            )

            for key, value in report.items():

                print(
                    f"{key}: {value:,}"
                )

            # --------------------------------------------------
            # Validation rules
            # --------------------------------------------------

            if any([

                report[
                    "trained_items_missing"
                ] != 0,

                report[
                    "catalog_items_not_in_vocab"
                ] != 0,

                report[
                    "duplicate_item_ids"
                ] != 0,

                report[
                    "null_item_ids"
                ] != 0,

                report[
                    "null_display_names"
                ] != 0,

            ]):

                raise ValueError(
                    "Catalog validation failed."
                )

            print(
                "\nCatalog validation: PASSED"
            )

            return report

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # SAVE
    # ==========================================================

    def save_catalog(
        self,
        catalog
    ):

        try:

            output_path = (
                self.config.output_catalog_path
            )

            output_directory = (
                os.path.dirname(
                    output_path
                )
            )

            if output_directory:

                os.makedirs(
                    output_directory,
                    exist_ok=True
                )

            catalog.to_parquet(
                output_path,
                index=False
            )

            logging.info(
                f"Catalog saved to: {output_path}"
            )

            return output_path

        except Exception as e:

            raise CustomeException(e, sys) from e

    # ==========================================================
    # MAIN
    # ==========================================================

    def initiate_catalog_build(self):

        try:

            catalog, item_to_idx = (
                self.build_catalog()
            )

            report = (
                self.validate_catalog(
                    catalog,
                    item_to_idx
                )
            )

            output_path = (
                self.save_catalog(
                    catalog
                )
            )

            print(
                "\n========================================"
            )

            print(
                "PRODUCT CATALOG BUILD COMPLETE"
            )

            print(
                "========================================"
            )

            print(
                f"Rows: {len(catalog):,}"
            )

            print(
                f"Output: {output_path}"
            )

            print(
                "\nColumns:"
            )

            for column in catalog.columns:

                print(
                    f"  - {column}"
                )

            return output_path, report

        except Exception as e:

            raise CustomeException(e, sys) from e


# ==============================================================
# SCRIPT ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    builder = CatalogBuilder()

    builder.initiate_catalog_build()
