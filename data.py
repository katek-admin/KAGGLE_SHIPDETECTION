import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json

from utils import rle_decode

import config

images_file = os.getcwd()+'/inputdata/train_images.npy'
masks_file = os.getcwd()+'/inputdata/masks_images.npy'

test_images_file = os.getcwd()+'/inputdata/test_images.npy'



# Function to preprocess the mask
def preprocess_mask(mask_rle):
    shape=(config.ORIGINAL_IMAGE_HEIGHT, config.ORIGINAL_IMAGE_WIDTH)
    target_size=(config.TARGET_IMAGE_HEIGHT, config.TARGET_IMAGE_WIDTH)

    if pd.isna(mask_rle):  # If no mask is provided, create a zero mask
        original_mask = np.zeros(shape, dtype=np.uint8)
    else:
        original_mask = rle_decode(mask_rle, shape)

    mask = tf.convert_to_tensor(original_mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, target_size, method = 'nearest')
    # Ensure the resized mask retains binary values (0.0 and 1.0)
    mask = tf.round(mask)
    return mask

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(config.TARGET_IMAGE_HEIGHT, config.TARGET_IMAGE_WIDTH)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size, method = 'nearest')
    return img

def load_data():
    images = []
    masks = []

    if os.path.exists(images_file):
        images = np.load(images_file)
    
    if os.path.exists(masks_file):
        masks = np.load(masks_file)

    if len(images)==0 or len(masks)==0:
        masks_df = pd.read_csv(os.getcwd()+config.MASK_CSV_PATH)

        for index, row in masks_df.iterrows():
            if index==config.IMAGES_TOINCLUDE:
                break
            img_file = os.path.join(config.TRAIN_IMAGES_PATH, row['ImageId'])
            if os.path.exists(img_file):
                img = load_and_preprocess_image(img_file)
                mask = preprocess_mask(row['EncodedPixels'])
                images.append(img)
                masks.append(mask)

        # Convert lists to numpy arrays
        images = np.array(images)
        masks = np.array(masks)


        np.save(images_file, images)
        np.save(masks_file, masks)

    return images, masks

# Example data loading function
def load_test_data():
    test_images = []

    if os.path.exists(test_images_file):
        test_images = np.load(images_file)
    
   
    if len(test_images)==0:
        index=0
        #for _, _, filenames in os.walk(config.TEST_IMAGES_PATH):
        filenames = os.listdir(config.TEST_IMAGES_PATH)
                              
        for filename in filenames:
            if index==10:
                break

            print(os.path.join(filename))
            img_file = os.path.join(config.TEST_IMAGES_PATH, filename)
            img = load_and_preprocess_image(img_file)
            test_images.append(img)
            index+=1

        # Convert lists to numpy arrays
        test_images = np.array(test_images)
        np.save(test_images_file, test_images)

    return test_images


images, masks = load_data()
X_test = load_test_data()

# Split the data
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=config.VALIDATION_SPLIT, random_state=42)

X_train = tf.data.Dataset.from_tensor_slices(X_train)
y_train = tf.data.Dataset.from_tensor_slices(y_train)

X_val = tf.data.Dataset.from_tensor_slices(X_val)
y_val = tf.data.Dataset.from_tensor_slices(y_val)

X_test = tf.data.Dataset.from_tensor_slices(X_test)

# Add labels to dataframe objects (one-hot-encoded)
train_dataset = tf.data.Dataset.zip((X_train, y_train))
val_dataset = tf.data.Dataset.zip((X_val, y_val))

# Apply the batch size to the dataset
batched_train_dataset = train_dataset.batch(config.BATCH_SIZE)
batched_val_dataset = val_dataset.batch(config.BATCH_SIZE)
batched_test_dataset = X_test.batch(config.BATCH_SIZE)

# Adding autotune for pre-fetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
batched_train_dataset = batched_train_dataset.prefetch(buffer_size=AUTOTUNE)
batched_val_dataset = batched_val_dataset.prefetch(buffer_size=AUTOTUNE)
batched_test_dataset = batched_test_dataset.prefetch(buffer_size=AUTOTUNE)

#check data
# View images and associated labels
for images, masks in batched_train_dataset.take(1):
    car_number = 0
    plt.figure(figsize=(12, 4))
    for image_slot in range(4):
        ax = plt.subplot(2, 2, image_slot + 1)
        if image_slot % 2 == 0:
            plt.imshow((images[car_number])) 
            class_name = 'Image'
        else:
            plt.imshow(masks[car_number], cmap = 'gray')
            print(masks[car_number].numpy())
            #plt.colorbar()
            class_name = 'Mask'
            car_number += 1            
        plt.title(class_name)
        plt.axis("off")
    plt.show()






