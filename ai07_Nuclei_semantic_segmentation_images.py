# -*- coding: utf-8 -*-
"""
Created on Wed July 27 14:28:51 2022

@author: MAKMAL2-PC23
"""

#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import io
import glob, os

#%%
#2. Data preparation
#load the train image and masks
train_dir = r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\project_ai07\data-science-bowl-2018-2\train"

images_train =[]
masks_train =[]

image_train_dir = os.path.join(train_dir, 'inputs')
for image_file in os.listdir(image_train_dir):
    image= cv2.imread(os.path.join(image_train_dir,image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(128,128))
    images_train.append(image)
    
masks_train_dir = os.path.join(train_dir, 'masks')
for mask_file in os.listdir(masks_train_dir):
    mask = cv2.imread(os.path.join(masks_train_dir,mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks_train.append(mask)
    
#%%
#load the test image and masks
test_dir = r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\project_ai07\data-science-bowl-2018-2\test"

images_test = []
masks_test = []

image_test_dir = os.path.join(test_dir, 'inputs')
for image_file in os.listdir(image_test_dir):
    image = cv2.imread(os.path.join(image_test_dir,image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(128,128))
    images_test.append(image)
    
masks_test_dir = os.path.join(test_dir, 'masks')
for mask_file in os.listdir(masks_test_dir):
    mask = cv2.imread(os.path.join(masks_test_dir,mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks_test.append(mask)
    
#%%
#3. Convert images and masks into numpy array
train_images_np = np.array(images_train)
train_masks_np = np.array(masks_train)
test_images_np = np.array(images_test)
test_masks_np = np.array(masks_test)

#%%
#4. Display some image examples
plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = images_train[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()   

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = masks_train[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show() 

#%%
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
train_masks_converted = np.ceil(train_masks_np_exp/255)
train_masks_converted = 1 - train_masks_converted


test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
test_masks_converted = np.ceil(test_masks_np_exp/255)
test_masks_converted = 1 - test_masks_converted

#%%
test_images_converted = test_images_np/255.0
train_images_converted = train_images_np/255.0

#%%
# Perform train test split
SEED = 42
x_train, x_test, y_train, y_test = train_test_split(train_images_converted,train_masks_converted,test_size=0.2,random_state=SEED)

#%%

x_train_tensor = tf.data.Dataset.from_tensor_slices(train_images_converted)
x_test_tensor = tf.data.Dataset.from_tensor_slices(test_images_converted)
y_train_tensor = tf.data.Dataset.from_tensor_slices(train_masks_converted)
y_test_tensor = tf.data.Dataset.from_tensor_slices(test_masks_converted)

#%%
train = tf.data.Dataset.zip((x_train_tensor, y_train_tensor))
test = tf.data.Dataset.zip((x_test_tensor, y_test_tensor))

#%%
# Create subclass for data augmentation
class Augment(layers.Layer):
  def __init__(self,seed=SEED):
    super().__init__()
    self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=SEED)
    self.augment_labels = layers.RandomFlip(mode='horizontal',seed=SEED)
  
  def call(self,inputs,labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs,labels 

#%%
# Preprocess the batch data
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEP_PER_EPOCH = 800 // BATCH_SIZE
VALIDATION_STEPS = 200 // BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size = AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%
#Create image segmentation model
base_model = tf.keras.applications.MobileNetV2(input_shape = [128,128,3], include_top = False)
#Define the pretrained as downsampler
layer_names = [
    'block_1_expand_relu',   
    'block_3_expand_relu',   
    'block_6_expand_relu',
    'block_13_expand_relu',  
    'block_16_project',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False
# Define the upsampler block
up_stack = [
    pix2pix.upsample(512, 3), 
    pix2pix.upsample(256, 3),  
    pix2pix.upsample(128, 3),  
    pix2pix.upsample(64, 3),   
]
#Create modified unet model
def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

# Downsampler
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

# Upsampler
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2
model = unet_model(output_channels= OUTPUT_CLASSES)

#%%
# Compile the model
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.summary()
#%%
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()
    
for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])
    
#%%

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)[0]])
        else:
            display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis,...]))[0]])
            
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\n Sample prediction after epoch {}\n".format(epoch+1))
        
import datetime

log_dir = r"C:\Users\User\Desktop\deep learning\log\tb_logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1, profile_batch = 0)

#%%
#Train the model
EPOCH = 20
history = model.fit(train, epochs = EPOCH, steps_per_epoch = STEP_PER_EPOCH, validation_steps= VALIDATION_STEPS, validation_data= test, callbacks=[DisplayCallback(), tb_callback]) 
#%%
#Test evaluation
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")

#%%
#Deploy model 
show_predictions(test,6)


