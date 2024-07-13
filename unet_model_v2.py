import tensorflow as tf
from keras import layers, optimizers
from keras import backend

from config import TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH

data_augmentation = tf.keras.Sequential([
        layers.RandomFlip(mode="horizontal", seed=42),
        layers.RandomRotation(factor=0.01, seed=42),
        layers.RandomContrast(factor=0.2, seed=42)
])

# Build the U-Net model
#avoid destructive max pooling layers and stick to using strided convolutions instead 

def unet_model_v2(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = data_augmentation(inputs)
    
    # Contracting path
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same", kernel_initializer='he_normal')(x) 
    x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x) 
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same", kernel_initializer='he_normal')(x) 
    x = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x) 
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu", kernel_initializer='he_normal')(x) 
    x = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    
    # Expanding path
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", kernel_initializer='he_normal', strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", kernel_initializer='he_normal', strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", kernel_initializer='he_normal', strides=2)(x)

    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs]) 
    
    return model


def dice_coefficient(y_true, y_pred):
    smooth = 10e-6
    y_true_flat = backend.flatten(y_true)
    y_pred_flat = backend.flatten(y_pred)
    intersection = backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (backend.sum(y_true_flat) + backend.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


model = unet_model_v2((TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH,3))
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, epsilon=1e-06), loss=[dice_loss], metrics=[dice_coefficient])


# TODO: delete
model.summary()

