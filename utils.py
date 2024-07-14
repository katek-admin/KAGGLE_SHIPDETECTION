import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  

def create_mask(pred_mask):
    mask = pred_mask[..., -1] >= 0.5
    pred_mask[..., -1] = tf.where(mask, 1, 0)
    # Return only first mask of batch
    return pred_mask[0]

def display(display_list):
    plt.figure(figsize=(15, 15))

    if len(display_list) == 2:
        title = ['Input Image', 'Predicted Mask']
    else:
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        dice_coef = tf.keras.backend.round(dice_coefficient(display_list[1], display_list[2])*100)/100

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        if i==2:
            plt.title(f'{title[i]} - dice coef: {dice_coef}')
        else:
            plt.title(f'{title[i]}')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show(block=True)

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    smooth = 10e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    bce = bce_loss(y_true, y_pred)
    return dice + bce

    