# %% [markdown]
# # MODEL 3

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
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
# Defining class names
labels = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']

# %%
num_row = len(labels) // 2
num_col = len(labels) // num_row

fig, axes = plt.subplots(num_row, num_col, figsize=(2 * num_row, 15 * num_col))
for i in range(len(labels)):
  if i < len(labels):
    class_imgs = next(os.walk('{}/{}/'.format(dataset_dir, labels[i])))[2]
    class_img = class_imgs[0]
    img = Image.open('{}/{}/{}'.format(dataset_dir, labels[i], class_img))
    ax = axes[i//num_col, i%num_col]
    ax.imshow(np.array(img))
    ax.set_title('{}'.format(labels[i]))
      
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### DATA GENERATOR CREATION

# %%
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,
                                        height_shift_range=50,
                                        width_shift_range=50,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='reflect',
                                        rescale=1/255,
                                        validation_split=0.3)

# %%
train_data_gen = image_generator.flow_from_directory(directory=dataset_dir,
                                                     subset='training')
val_data_gen = image_generator.flow_from_directory(directory=dataset_dir,
                                                   subset='validation')

# %%
print("Assigned labels")
print(train_data_gen.class_indices)
print()
print("Target classes")
print(val_data_gen.classes)

# %% [markdown]
# ##### PRINT FROM BATCH

# %%
# Get sample batch
batch = next(train_data_gen)[0]

# Get and print first image of the third batch
image = batch[3]

fig = plt.figure(figsize=(6, 4))
plt.imshow(np.uint8(image*255))

# Print input dimension
input_shape = next(train_data_gen)[0].shape[1:]
print(input_shape)

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
# ##### EPOCH DECLARATION

# %%
# Originally trained in 50 epochs
# epochs = 50

epochs = 3

# %% [markdown]
# ##### MODEL DEFINITION

# %%
def build_model(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    resize_layer = tfkl.Resizing(64, 64, interpolation="bicubic")(input_layer)

    conv1 = tfkl.Conv2D(
        filters = 16,
        kernel_size=(3, 3),
        strides = (1, 1),
        padding = 'same',
        activation = 'relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(resize_layer)
    pool1 = tfkl.MaxPooling2D(
        pool_size = (2, 2)
    )(conv1)

    conv2 = tfkl.Conv2D(
        filters = 32,
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

    flattening_layer = tfkl.Flatten(name='Flatten')(pool4)
    flattening_layer = tfkl.Dropout(0.3, seed=seed)(flattening_layer)

    classifier_layer = tfkl.Dense(
        units= 256 ,
        activation='relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed),
        kernel_regularizer = tfk.regularizers.l2(1e-5),
        name='Classifier')(flattening_layer)
    classifier_layer_f = tfkl.Dropout(0.3, seed=seed, name='ClassifierDropout')(classifier_layer)

    output_layer = tfkl.Dense(
        units=14,
        activation='softmax',
        kernel_initializer = tfk.initializers.GlorotUniform(seed),
        kernel_regularizer = tfk.regularizers.l2(1e-5),
        name='Output')(classifier_layer_f)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])

    # Return the model
    return model

# %%
model = build_model(input_shape)
model.summary()

# %% [markdown]
# ##### BATCH SIZE DECLARATION

# %%
batch_size = 32

# %% [markdown]
# ##### MODEL FIT

# %%
# callbacks = create_folders_and_callbacks(model_name = "Model_less_layer")

# Train the models
standard_history = model.fit(
    x = train_data_gen,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = val_data_gen,
    # callbacks = callbacks -> ok without sample case
).history

# %% [markdown]
# ##### OUTPUT GRAPHS

# %%
plt.figure(figsize=(15,5))
plt.plot(standard_history['loss'], label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(standard_history['val_loss'], label='Validation', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Categorical Crossentropy')
plt.grid(alpha=.3)

plt.figure(figsize=(15,5))
plt.plot(standard_history['accuracy'], label='Training', alpha=.8, color='#ff7f0e', linestyle='--')
plt.plot(standard_history['val_accuracy'], label='Validation', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)
