"""
Feature Engineering Module for FinTech Guardian (Phase 2)
Author: Renita Parumal
Date: September 2026

This script executes context-aware feature engineering including financial strain
ratios, cyclical load-shedding encodings, leakage-safe regional benchmarks,
zero-inflated transformations, and multicollinearity filtering.
"""

import os
import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create business-intelligent features with ethical safeguards."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_metadata = {}
        self.initial_features = list(self.df.columns)
        self.created_features = []

    def create_financial_strain_ratio(self) -> None:
        """Calculate debt-to-income ratio with safe zero handling."""
        # Division by zero handled by setting ratio to 0.0 when income <= 0
        income = self.df["income"].fillna(0)
        debt = self.df["debt"].fillna(0)

        self.df["debt_to_income"] = np.where(income > 0, debt / income, 0.0)

        # Cap extreme outliers at 99th percentile for numerical stability
        cap_val = self.df["debt_to_income"].quantile(0.99)
        self.df["debt_to_income"] = np.clip(
            self.df["debt_to_income"], 0, cap_val
        )

        self.created_features.append("debt_to_income")
        self.feature_metadata["debt_to_income"] = (
            "Identifies customers at risk of default due to overcommitment. "
            "Safe ratio computed with zero-income guardrails."
        )

    def encode_load_shedding_impact(self) -> None:
        """Convert load shedding hours to cyclical features (captures weekly patterns)."""
        # Load shedding hours cycled on a 168-hour (weekly) period
        hours = self.df["load_shedding_hours"].fillna(0)
        period = 168.0

        self.df["load_shedding_sin"] = np.sin(2 * np.pi * hours / period)
        self.df["load_shedding_cos"] = np.cos(2 * np.pi * hours / period)

        self.created_features.extend(
            ["load_shedding_sin", "load_shedding_cos"]
        )
        self.feature_metadata["load_shedding_sin"] = (
            "Preserves circular relationship: Friday 22h close to Saturday 02h (Sine component)"
        )
        self.feature_metadata["load_shedding_cos"] = (
            "Preserves circular relationship: Friday 22h close to Saturday 02h (Cosine component)"
        )

    def regional_benchmarks(self, train_stats: pd.DataFrame = None) -> None:
        """Add region-level aggregates WITHOUT data leakage."""
        if train_stats is None:
            # Training mode: Compute stats ONLY on current dataset
            group_stats = (
                self.df.groupby("region")["income"]
                .agg(
                    region_mean_income="mean",
                    region_count="count",
                )
                .reset_index()
            )

            # Check for low sample regions and document warning
            low_sample_regions = group_stats[group_stats["region_count"] < 50][
                "region"
            ].tolist()
            if low_sample_regions:
                self.feature_metadata["region_mean_income_warning"] = (
                    f"Warning: Regions with <50 samples detected: {low_sample_regions}"
                )

            self.train_stats = group_stats.drop(columns=["region_count"])
        else:
            self.train_stats = train_stats

        # Merge leakage-safe regional statistics
        self.df = self.df.merge(self.train_stats, on="region", how="left")
        if "region_mean_income" not in self.created_features:
            self.created_features.append("region_mean_income")

        self.feature_metadata["region_mean_income"] = (
            "Regional economic baseline used to contextualize customer relative income without leakage."
        )

    def handle_zero_inflated_support_tickets(self) -> None:
        """Two-part transformation for zero-inflated support_tickets feature."""
        tickets = self.df["support_tickets"].fillna(0)

        # Part 1: Binary indicator (Has customer ever raised a ticket?)
        self.df["has_support_tickets"] = np.where(tickets > 0, 1, 0)

        # Part 2: Log-transformed intensity for non-zero counts
        self.df["support_tickets_log"] = np.where(
            tickets > 0, np.log1p(tickets), 0.0
        )

        self.created_features.extend(
            ["has_support_tickets", "support_tickets_log"]
        )
        self.feature_metadata["has_support_tickets"] = (
            "Binary indicator for ticket generation to isolate zero-inflated baseline."
        )
        self.feature_metadata["support_tickets_log"] = (
            "Log-1p transformed ticket severity for active support users."
        )

    def remove_multicollinearity(self, threshold: float = 0.85) -> None:
        """Validation safeguard: Drop features with correlation higher than threshold."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > threshold)
        ]

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            for feat in to_drop:
                if feat in self.created_features:
                    self.created_features.remove(feat)

    def calculate_max_vif(self) -> float:
        """Compute maximum Variance Inflation Factor across numeric features."""
        numeric_cols = self.df.select_dtypes(
            include=[np.number]
        ).dropna(axis=1)
        if numeric_cols.shape[1] < 2:
            return 1.0

        corr_matrix = numeric_cols.corr().values
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            vif_vals = np.diag(inv_corr)
            return float(np.nanmax(vif_vals))
        except np.linalg.LinAlgError:
            return 1.0

    def __str__(self) -> str:
        """Return formatted feature engineering summary string required by specifications."""
        num_new = len(self.created_features)
        max_vif = self.calculate_max_vif()
        return f"Created {num_new} new features | Max VIF: {max_vif:.1f}"

    def run_full_engineering(self) -> pd.DataFrame:
        """Execute all feature engineering steps seamlessly."""
        try:
            self.create_financial_strain_ratio()
            self.encode_load_shedding_impact()
            self.regional_benchmarks()
            self.handle_zero_inflated_support_tickets()
            self.remove_multicollinearity(threshold=0.85)
            return self.df
        except Exception as e:
            raise ValueError(f"Feature engineering failed: {str(e)}") from e


if __name__ == "__main__":
    # Ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    input_path = "data/processed/cleaned_customers.csv"
    output_path = "data/processed/engineered_features.csv"

    # Dummy data generator if input file does not exist locally yet
    if not os.path.exists(input_path):
        dummy_df = pd.DataFrame(
            {
                "customer_id": range(100),
                "income": np.random.uniform(0, 50000, 100),
                "debt": np.random.uniform(0, 20000, 100),
                "load_shedding_hours": np.random.randint(0, 168, 100),
                "region": np.random.choice(
                    ["Gauteng", "KZN", "Western Cape"], 100
                ),
                "support_tickets": np.random.choice([0, 0, 0, 1, 3, 5], 100),
                "township_flag": np.random.choice([0, 1], 100),
            }
        )
        dummy_df.to_csv(input_path, index=False)

    # Execute pipeline
    raw_df = pd.read_csv(input_path)
    engineer = FeatureEngineer(raw_df)
    engineered_df = engineer.run_full_engineering()

    # Save output dataset
    engineered_df.to_csv(output_path, index=False)

    print("Success: Feature Engineering Complete!")
    print(engineer)
