# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:poli_challenges]
#     language: python
#     name: conda-env-poli_challenges-py
# ---

# %% [markdown]
# ##### IMPORTS

# %%
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from PIL import Image
from pathlib import Path

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# %% [markdown]
# ##### SEED DEFINITION

# %%
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# %% [markdown]
# ##### DATASET DIR

# %%
root = Path.cwd()

# Full dataset
dataset_dir = root / "data/TrainingCH2.csv"

# Sample dataset
# dataset_dir = root / "data/TrainingCH2_sample.csv"

print(dataset_dir)

# %%
dataset = pd.read_csv(dataset_dir)
print(dataset.shape)
dataset.head()

# %%
dataset.info()


# %%
def inspect_dataframe(df, columns):
    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(df[col])
        axs[i].set_title(col)
    plt.show()
    
inspect_dataframe(dataset, dataset.columns)

# %% [markdown]
# ##### WINDOW DEFINITION

# %%
window = 300

# %% [markdown]
# ##### AUTOCORRELATION ANALYSIS

# %%
# Min-Max per column on TRAIN; safe denominator for zero-range columns
X_min = dataset.min()
X_max = dataset.max()
denom = (X_max - X_min).replace(0, 1.0)

def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df - X_min) / denom

train_scaled = scale_df(dataset)

# %%
max_lag = 200
num_cols = min(4, train_scaled.shape[1])  # plot a few columns for readability
cols_to_plot = list(train_scaled.columns[:num_cols])
# 
fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3 * len(cols_to_plot)))
if len(cols_to_plot) == 1:
    axes = [axes]

for ax, col in zip(axes, cols_to_plot):
    x = train_scaled[col].values
    acf = [pd.Series(x).autocorr(lag=l) for l in range(1, max_lag + 1)]
    ax.stem(range(1, max_lag + 1), acf)
    ax.axvline(window, linestyle="--")  # show current window
    ax.set_title(f"ACF - {col} (showing lag 1..{max_lag})")
    ax.set_xlabel("lag"); ax.set_ylabel("corr")
    
plt.tight_layout()
plt.show()

# %% [raw]
# The ACF analysis suggest us that a correct value for windows could be sampled from this range: 75 - 90
