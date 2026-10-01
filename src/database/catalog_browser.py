"""
src/services/catalog_browser.py

Read-only catalog access for the Streamlit store/browse pages.
Deliberately NOT a recommendation system -- no scoring, no ranking,
just listing and lookup against the existing product_catalog.parquet.
RecommendationService and PredictionPipeline remain the only sources
of actual recommendations.
"""

import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomeException


class CatalogBrowser:

    def __init__(self, catalog_path="product_catalog.parquet"):
        try:
            self.catalog = pd.read_parquet(catalog_path)
            logging.info(f"CatalogBrowser loaded {len(self.catalog):,} products")
        except Exception as e:
            raise CustomeException(e, sys) from e

    def list_products(self, limit=12, category_id=None, search=None):
        try:
            df = self.catalog

            if category_id is not None:
                df = df[df["category_id"] == category_id]

            if search:
                df = df[df["display_name"].str.contains(search, case=False, na=False)]

            return df.head(limit).to_dict("records")

        except Exception as e:
            raise CustomeException(e, sys) from e

    def get_product(self, retailrocket_item_id):
        try:
            row = self.catalog[self.catalog["retailrocket_item_id"] == retailrocket_item_id]
            if row.empty:
                return None
            return row.iloc[0].to_dict()
        except Exception as e:
            raise CustomeException(e, sys) from e

    def list_categories(self, limit=30):
        try:
            categories = self.catalog["category_id"].dropna().unique().tolist()
            return sorted(categories)[:limit]
        except Exception as e:
            raise CustomeException(e, sys) from e
