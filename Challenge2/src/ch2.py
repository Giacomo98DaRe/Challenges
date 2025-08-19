# -*- coding: utf-8 -*-
# %% [markdown]
# # CHALLENGE 2

# %% [markdown]
# ##### IMPORTS

# %%
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

tfk = tf.keras
tfkl = tf.keras.layers
tf.get_logger().setLevel('ERROR')
print(tf.__version__)

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ##### SEED DECLARATION

# %%
# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# %% [markdown]
# ##### DATASET PATH

# %%
root = Path.cwd()

# Full dataset
dataset_dir = root / "data/TrainingCH2.csv"

# Sample dataset
# dataset_dir = root / "data/TrainingCH2_sample.csv"

print(dataset_dir)

# %%
dataset = pd.read_csv(dataset_dir)

# %% [markdown]
# ##### VALIDATION SET

# %%
train_ratio = 0.8
split_idx = int(len(dataset) * train_ratio)

train_df = dataset.iloc[:split_idx].copy()
val_df   = dataset.iloc[split_idx:].copy()

# %%
# Min-Max per column on TRAIN; safe denominator for zero-range columns
X_min = train_df.min()
X_max = train_df.max()
denom = (X_max - X_min).replace(0, 1.0)

def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df - X_min) / denom

train_scaled = scale_df(train_df)
val_scaled   = scale_df(val_df)


# %% [markdown]
# ##### SEQUENCE

# %%
def build_sequences(df: pd.DataFrame, target_labels: list[str] | None, window: int, stride: int, telescope: int):
    arr = df.values
    if target_labels is None:
        target_cols = list(df.columns)  # multivariate target = all columns
    else:
        target_cols = list(target_labels)
    y_arr = df[target_cols].values

    X_list, y_list = [], []
    last_start = len(df) - window - telescope
    if last_start < 0:
        return np.empty((0, window, df.shape[1])), np.empty((0, telescope, len(target_cols)))
    for start in range(0, last_start + 1, stride):
        X_list.append(arr[start : start + window])
        y_list.append(y_arr[start + window : start + window + telescope])
    return np.asarray(X_list), np.asarray(y_list)

target_labels = list(dataset.columns)  # multivariate forecasting (all columns)

# %%
target_labels = dataset.columns

telescope = 30 # number of data to predict
window = 80 # number of data where predict from
stride = 10 # window shift

X_train, y_train = build_sequences(
    train_scaled, target_labels, window=window, stride=stride, telescope=telescope
)
X_val, y_val = build_sequences(
    val_scaled, target_labels, window=window, stride=stride, telescope=telescope
)

print("Train shapes:", X_train.shape, y_train.shape)
print("Val   shapes:", X_val.shape, y_val.shape)

# %% [markdown]
# ##### DATA TO PREDICT

# %%
future = val_df.iloc[-window:].copy()
future = (future - X_min) / denom
future = np.expand_dims(future.values, axis=0)

print("future shape:", future.shape)  # (1, window, n_features)

# %% [markdown]
# ##### TASK VISUALIZATION

# %%
def inspect_multivariate(X, y, columns, telescope, idx=None):
    if(idx==None):
        idx=np.random.randint(0,len(X))
        
    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(np.arange(len(X[0,:,i])), X[idx,:,i])
        axs[i].scatter(np.arange(len(X[0,:,i]), len(X_train[0,:,i])+telescope), y[idx,:,i], color='orange')
        axs[i].set_title(col)
        axs[i].set_ylim(0,1)
    plt.show()

inspect_multivariate(X_train, y_train, target_labels, telescope)

# %% [markdown]
# ##### EPOCH AND I/O SHAPE

# %%
input_shape = X_train.shape[1:]
output_shape = y_train.shape[1:]
batch_size = 64

print(input_shape)
print(output_shape)

# epochs = 50
epochs = 3

# %% [markdown]
# ##### MODEL DEFINITION

# %%
def build_CONV_LSTM_model(input_shape, output_shape):
    
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    convlstm = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True))(input_layer)
    convlstm = tfkl.Conv1D(128, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.MaxPool1D()(convlstm)
    convlstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(convlstm)
    convlstm = tfkl.Conv1D(256, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.GlobalAveragePooling1D()(convlstm)
    convlstm = tfkl.Dropout(.5)(convlstm)

    # In order to predict the next values for more than one channel,
    # we can use a Dense layer with a number given by telescope*num_channels,
    # followed by a Reshape layer to obtain a tensor of dimension
    # [None, telescope, num_channels]
    dense = tfkl.Dense(output_shape[-1]*output_shape[-2], activation='relu')(convlstm)
    output_layer = tfkl.Reshape((output_shape[-2],output_shape[-1]))(dense)
    output_layer = tfkl.Conv1D(output_shape[-1], 1, padding='same')(output_layer)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(), metrics=['mae'])

    # Return the model
    return model

# %%
model = build_CONV_LSTM_model(input_shape, output_shape)
model.summary()

# tfk.utils.plot_model(model, expand_nested=True)

# %% [markdown]
# ##### TRAINING MODEL

# %%
# Train the model

history = model.fit(
    x = X_train,
    y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
        tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)
    ],
    validation_data = (X_val, y_val)
).history

# %% [markdown]
# ##### OUTPUT GRAPHS

# %%
best_epoch = np.argmin(history['val_loss'])
plt.figure(figsize=(17,4))
plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
plt.title('Mean Squared Error (Loss)')
plt.legend()
plt.grid(alpha=.3)
plt.show()

# %%
plt.figure(figsize=(17,4))
plt.plot(history['mae'], label='Training accuracy', alpha=.8, color='#ff7f0e')
plt.plot(history['val_mae'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
plt.title('Mean Absolute Error')
plt.legend()
plt.grid(alpha=.3)
plt.show()
