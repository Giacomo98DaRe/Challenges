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

# %% [markdown] colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 302, "status": "ok", "timestamp": 1637878131906, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="fW9HGza-0KYs" outputId="cf8da68e-b311-4ed8-92d6-f754286cd1da"
# # DATASET ANALYSIS 

# %% [markdown]
# ##### IMPORTS

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 416, "status": "ok", "timestamp": 1637880791266, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="uzF8nMLW0whb" outputId="748e154c-a9d3-4a59-80d7-67f2f4e64ecc"
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from PIL import Image
from pathlib import Path

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# %% [markdown]
# ##### SEED AND DATASET PATH EXTRACTION

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1637880791675, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="vBcLsZevTSlr"
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1637878134371, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="QlVEeZUY1tJ0" outputId="07881823-dbf3-4b0f-a97d-d125d6e34478"
ROOT = Path.cwd()

# Full dataset
# dataset_dir = ROOT / "../data/FullPlantDataset/training"

# Sample dataset
dataset_dir = ROOT / "../data/FullPlantDataset/training_sample"

print(dataset_dir)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1637878134371, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="QlVEeZUY1tJ0" outputId="07881823-dbf3-4b0f-a97d-d125d6e34478"
class_name = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']

# %% [markdown]
# ##### DATASET SIZE

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator for training, validation, and test sets
train_data_gen = ImageDataGenerator()

# Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
train_gen = train_data_gen.flow_from_directory(directory=dataset_dir,
                                               target_size=(256,256),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=True,
                                               seed=seed)

# %% [markdown]
# ##### DATASET DISTRIBUTION

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4255, "status": "ok", "timestamp": 1637878193205, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="QsE7BxZ4Elsd" outputId="c99ec094-00cf-44a7-f904-2e98f16da16e"
values = []
from pathlib import Path

for label in class_name: 
  plant_path = dataset_dir / label
  initial_count = 0
  for path in Path(plant_path).iterdir():
      if path.is_file():
        initial_count += 1
  values.append(initial_count)      

print(values)  

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"elapsed": 313, "status": "ok", "timestamp": 1637878953930, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="aNWxkaRx1Jma" outputId="e2c634ac-4a6d-4459-acf7-5cbb4dd44562"
plt.figure(figsize=(15,5))
plt.bar(class_name, values)
plt.xticks(rotation=45); plt.xlabel("Class"); plt.ylabel("Number of samples"); plt.title("Class distribution"); plt.tight_layout()
plt.show()

# %%
# Relative percentage of the lables in the dataset
total = sum(values)
percent_by_label = {lbl: (v / total) * 100 if total else 0.0
                    for lbl, v in zip(class_name, values)}
print(percent_by_label)

# %% [markdown]
# ##### DATASET EXAMPLES

# %%
# Grid of examples from dataset
cols = 7
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = np.atleast_1d(axes).ravel()  

labels = list(class_name)

for i, lab in enumerate(labels):
    ax = axes[i]
    plant_path = dataset_dir / lab
    first_file = sorted([p for p in plant_path.iterdir() if p.is_file()])[0]
    with Image.open(first_file) as im:
        ax.imshow(im)
    ax.set_title(lab, fontsize=10)
    ax.axis("off")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ##### DATASET AUGMENTATION EXAMPLES

# %% colab={"base_uri": "https://localhost:8080/", "height": 364} executionInfo={"elapsed": 2583, "status": "ok", "timestamp": 1637880876910, "user": {"displayName": "Giacomo Da Re", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "03451881176407058316"}, "user_tz": -60} id="zyVttE_wKwJx" outputId="403ec601-7561-4c86-ada7-b1888406b63b"
# Get sample image
image = next(train_gen)[0][7]

# Create an instance of ImageDataGenerator for each transformation
rot_gen = ImageDataGenerator(rotation_range=30)
shift_gen = ImageDataGenerator(width_shift_range=50)
zoom_gen = ImageDataGenerator(zoom_range=0.3)
flip_gen = ImageDataGenerator(horizontal_flip=True)

# Get random transformations
rot_t = rot_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Rotation:', rot_t, '\n')
shift_t = shift_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Shift:', shift_t, '\n')
zoom_t = zoom_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Zoom:', zoom_t, '\n')
flip_t = flip_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Flip:', flip_t, '\n')

# Apply the transformation
gen = ImageDataGenerator(fill_mode='constant', cval=0.)
rotated = gen.apply_transform(image, rot_t)
shifted = gen.apply_transform(image, shift_t) 
zoomed = gen.apply_transform(image, zoom_t) 
flipped = gen.apply_transform(image, flip_t)  

# Plot original and augmented images
fig, ax = plt.subplots(1, 5, figsize=(15, 45))
ax[0].imshow(np.uint8(image))
ax[0].set_title('Original')
ax[1].imshow(np.uint8(rotated))
ax[1].set_title('Rotated')
ax[2].imshow(np.uint8(shifted))
ax[2].set_title('Shifted')
ax[3].imshow(np.uint8(zoomed))
ax[3].set_title('Zoomed')
ax[4].imshow(np.uint8(flipped))
ax[4].set_title('Flipped')
