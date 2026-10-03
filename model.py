"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
import os, tempfile, tarfile, urllib.request
import pandas as pd

def load_housing():
    # TODO: Download housing.tgz once into tempfile.gettempdir() and read housing/housing.csv from it.
    p = os.path.join(tempfile.gettempdir(), "housing.tgz")
    if not os.path.exists(p):
        urllib.request.urlretrieve("https://github.com/ageron/data/raw/main/housing.tgz", p)
    with tarfile.open(p) as t:
        return pd.read_csv(t.extractfile("housing/housing.csv"))

# Step 2 - income_categories
import pandas as pd
import numpy as np
def income_categories(df: pd.DataFrame):
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    return pd.cut(df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5]).astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split
def stratified_split(df, test_size=0.2, random_state=42):
    # TODO: train_test_split stratified on income_categories(df); return (train_set, test_set).
    y = income_categories(df)
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state, stratify=y)
    return (train_set,test_set)

# Step 4 - explore_correlations
def explore_correlations(df):
    # TODO: Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.
    return df.corr(numeric_only=True)["median_house_value"].drop("median_house_value").sort_values(ascending=False)

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    new_df = df.copy()
    new_df['rooms_per_house'] = new_df['total_rooms'] / new_df['households']  
    new_df['bedrooms_ratio'] = new_df['total_bedrooms']/ new_df['total_rooms']
    new_df['people_per_house'] = new_df['population'] / new_df['households']
    return new_df

# Step 6 - split_features_labels (not yet solved)
# TODO: implement

# Step 7 - ClusterSimilarity (not yet solved)
# TODO: implement

# Step 8 - numeric_pipeline (not yet solved)
# TODO: implement

# Step 9 - categorical_pipeline (not yet solved)
# TODO: implement

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

