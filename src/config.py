"""Configuration constants for the NBA injury-risk prediction project."""

# Analysis period: 2013-14 through 2018-19
FIRST_SEASON = 2013
LAST_SEASON = 2018

# Tracking data used in the project begins with the 2013-14 season
TRACKING_DATA_START = 2013

# Temporal modeling split
# Features from 2013-14 through 2016-17 are used for training.
# Features from 2017-18 form the held-out test set and predict 2018-19 outcomes.
TRAIN_SEASONS = list(range(2013, 2017))
TEST_SEASONS = [2017]

# Project-relative data paths
RAW_ELAP_DIR = "data/raw/elap733"
RAW_NBA_API_DIR = "data/raw/nba_api"
PROCESSED_DIR = "data/processed"

# Final regression target:
# number of injury-report appearances in the following season
TARGET_COL = "target_next_season"

# Random seed for reproducibility
RANDOM_SEED = 42
