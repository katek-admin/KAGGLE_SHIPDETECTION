import os
import pandas as pd

import config

images_file = os.getcwd()+'/inputdata/train_images.npy'
masks_file = os.getcwd()+'/inputdata/masks_images.npy'
test_images_file = os.getcwd()+'/inputdata/test_images.npy'

def masks_loading():
    shape=(config.ORIGINAL_IMAGE_HEIGHT, config.ORIGINAL_IMAGE_WIDTH)

    masks_df = pd.read_csv(os.getcwd()+config.MASK_CSV_PATH)

    masks_df = masks_df.sort_values(by=['ImageId'])
    print(masks_df)

    return masks_df["EncodedPixels"]


X_train = sorted([os.path.join(config.TRAIN_IMAGES_PATH, filename) for filename in os.listdir(config.TRAIN_IMAGES_PATH)], key=lambda x: x.split('/')[-1].split('.')[0])
y_train = masks_loading()

X_train = X_train[:config.IMAGES_TOINCLUDE]
y_train = y_train[:config.IMAGES_TOINCLUDE]

