import tensorflow as tf

from keras import layers, optimizers
from keras import backend

def conv_block(inputs=None, n_filters=64, dropout_prob=0, max_pooling=True):
    conv = layers.Conv2D(n_filters,  
                  3,   
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = layers.Conv2D(n_filters,  
                  3,   
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = layers.MaxPooling2D(pool_size=(2, 2))(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

data_augmentation = tf.keras.Sequential([
        layers.RandomFlip(mode="horizontal", seed=42),
        layers.RandomRotation(factor=0.01, seed=42),
        layers.RandomContrast(factor=0.2, seed=42)
])

def upsampling_block(expansive_input, contractive_input, n_filters=64):
    up = layers.Conv2DTranspose(
        n_filters,    
        3,    
        strides=(2, 2),
        padding='same',
        kernel_initializer='he_normal')(expansive_input)

    # Merge the previous output and the contractive_input
    merge = layers.concatenate([up, contractive_input], axis=3)

    conv = layers.Conv2D(n_filters,   
                  3,     
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = layers.Conv2D(n_filters,   
                  3,     
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv

def unet_model(input_size=(256, 256, 3), n_filters=64, n_classes=1):
    inputs = layers.Input(input_size)
    
    inputs = data_augmentation(inputs)

    # Contracting Path (encoding)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)

    # Bottleneck Layer
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)
    
    # Expanding Path (decoding)
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters*8)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    conv9 = layers.Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = layers.Conv2D(n_classes, 1, padding='same', activation="sigmoid")(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

def dice_coefficient(y_true, y_pred):
    smooth = 10e-6
    y_true_flat = backend.flatten(y_true)
    y_pred_flat = backend.flatten(y_pred)
    intersection = backend.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (backend.sum(y_true_flat) + backend.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


model = unet_model()
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, epsilon=1e-06), loss=[dice_loss], metrics=[dice_coefficient])


# TODO: delete
model.summary()