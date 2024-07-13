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
def unet_model(input_shape):
    inputs = tf.keras.Input(input_shape)
    inputs = data_augmentation(inputs)
    
    # Encoder (contracting path)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = layers.Dropout(0.3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = layers.Dropout(0.3)(conv5)

    # Decoder (expansive path)
    upconv4 = layers.Conv2DTranspose(512, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    upconv4 = layers.concatenate([upconv4, conv4], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv4)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    upconv3 = layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    upconv3 = layers.concatenate([upconv3, conv3], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    upconv2 = layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    upconv2 = layers.concatenate([upconv2, conv2], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv2)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    upconv1 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    upconv1 = layers.concatenate([upconv1, conv1], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv1)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    conv = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(conv)  # Output with sigmoid activation for binary segmentation
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def dice_coefficient(y_true, y_pred):
    smooth = 10e-6
    y_true_flat = backend.flatten(y_true)
    y_pred_flat = backend.flatten(y_pred)
    intersection = backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (backend.sum(y_true_flat) + backend.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    bce = bce_loss(y_true, y_pred)
    return dice + bce

model = unet_model((TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH,3))
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, epsilon=1e-06), loss=combined_loss, metrics=[dice_coefficient])


# TODO: delete
model.summary()

