# %% [markdown]
# # MODEL 1

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from PIL import Image
from pathlib import Path

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

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
ROOT = Path.cwd()

# Full dataset
# dataset_dir = ROOT / "../data/FullPlantDataset/training"

# Sample dataset
dataset_dir = ROOT / "../data/FullPlantDataset/training_sample"

print(dataset_dir)

# %% [markdown]
# ##### DEFINING CLASS NAMES

# %%
class_name = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']

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
# ##### TRAIN AND VALIDATION SPLIT

# %%
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = 0.2)
train_data_gen = image_generator.flow_from_directory(directory = dataset_dir,
                                                     subset = 'training')
val_data_gen = image_generator.flow_from_directory(directory = dataset_dir,
                                                   subset = 'validation')

# %%
print("Assigned labels")
print(train_data_gen.class_indices)
print()
print("Target classes")
print(val_data_gen.classes)

# %%
input_shape = (256, 256, 3)

# %% [markdown]
# ##### EPOCH DECLARATION

# %%
# Originally trained in 50 epochs
# epochs = 50

epochs = 3

# %% [markdown]
# ##### EARLY STOPPING

# %%
# Of course, in sample case it won't be effective
early_stopping = tfk.callbacks.EarlyStopping (monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = True)


# %% [markdown]
# ##### MODEL DEFINITION

# %%
# Model used:
# (Conv + ReLU + MaxPool) x 5 + FC x 2

def build_model(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    conv1 = tfkl.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides = (1, 1),
        padding = 'same',
        activation = 'relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(input_layer)
    pool1 = tfkl.MaxPooling2D(
        pool_size = (2, 2)
    )(conv1)

    conv2 = tfkl.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides = (1, 1),
        padding = 'same',
        activation = 'relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(pool1)
    pool2 = tfkl.MaxPooling2D(
        pool_size = (2, 2)
    )(conv2)

    conv3 = tfkl.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides = (1, 1),
        padding = 'same',
        activation = 'relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(pool2)
    pool3 = tfkl.MaxPooling2D(
        pool_size = (2, 2)
    )(conv3)

    conv4 = tfkl.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides = (1, 1),
        padding = 'same',
        activation = 'relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(pool3)
    pool4 = tfkl.MaxPooling2D(
        pool_size = (2, 2)
    )(conv4)

    conv5 = tfkl.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides = (1, 1),
        padding = 'same',
        activation = 'relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(pool4)
    pool5 = tfkl.MaxPooling2D(
        pool_size = (2, 2)
    )(conv5)

    flattening_layer = tfkl.Flatten(name='Flatten')(pool5)
    flattening_layer = tfkl.Dropout(0.3, seed=seed)(flattening_layer)

    classifier_layer = tfkl.Dense(units=512, name='Classifier', kernel_initializer=tfk.initializers.GlorotUniform(seed), kernel_regularizer = tfk.regularizers.l2(1e-4), activation='relu')(flattening_layer)
    classifier_layer = tfkl.Dropout(0.3, seed=seed)(classifier_layer)
    output_layer = tfkl.Dense(units=14, activation='softmax', kernel_initializer=tfk.initializers.GlorotUniform(seed), kernel_regularizer = tfk.regularizers.l2(1e-4), name='Output')(classifier_layer)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])

    # Return the model
    return model

# %%
# Build model (for NO augmentation training)
model = build_model(input_shape)
model.summary()

# %% [markdown]
# ##### CALLBACKS

# %%
# Utility function to create folders and callbacks for training
from datetime import datetime

def create_folders_and_callbacks(model_name):

  exps_dir = os.path.join('data_augmentation_experiments')
  if not os.path.exists(exps_dir):
      os.makedirs(exps_dir)

  now = datetime.now().strftime('%b%d_%H-%M-%S')

  exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
  if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)

  callbacks = []

  # Model checkpoint
  # ----------------
  ckpt_dir = os.path.join(exp_dir, 'ckpts')
  if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'),
                                                     save_weights_only=False, # True to save only weights
                                                     save_best_only=False) # True to save only the best epoch
  callbacks.append(ckpt_callback)

  # Visualize Learning on Tensorboard
  # ---------------------------------
  tb_dir = os.path.join(exp_dir, 'tb_logs')
  if not os.path.exists(tb_dir):
      os.makedirs(tb_dir)

  # By default shows losses and metrics for both training and validation
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                               profile_batch=0,
                                               histogram_freq=1)  # if > 0 (epochs) shows weights histograms
  callbacks.append(tb_callback)

  # Early Stopping
  # --------------
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights=True)
  callbacks.append(es_callback)

  return callbacks

# %% [markdown]
# ##### CLASS WEIGHTS CALCULUS

# %%
# Useless in case of sample dataset

from sklearn.utils.class_weight import compute_class_weight

y = train_data_gen.classes                  
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight = dict(zip(classes.tolist(), weights.astype(float).tolist()))

print(class_weight)

# %% [markdown]
# ##### MODEL FIT

# %%
# Out of scope now, but useful to have
# run_dir, callbacks = create_folders_and_callbacks(base_dir="runs", monitor="val_loss")

# Train the model
history = model.fit(
    x = train_data_gen,
    epochs = epochs,
    validation_data = val_data_gen,
    callbacks=[early_stopping],
    class_weight = class_weight
).history

# %% [markdown]
# ##### OUTPUT GRAPHS

# %%
plt.figure(figsize=(15,5))
plt.plot(history['loss'], label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(history['val_loss'], label='Validation', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Categorical Crossentropy')
plt.grid(alpha=.3)

plt.figure(figsize=(15,5))
plt.plot(history['accuracy'], label='Training', alpha=.8, color='#ff7f0e', linestyle='--')
plt.plot(history['val_accuracy'], label='Validation', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)
