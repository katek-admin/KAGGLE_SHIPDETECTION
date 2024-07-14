import tensorflow as tf
from keras import optimizers
import os

import config, data, utils
from unet_model import model



model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, epsilon=1e-06), loss=[utils.combined_loss], metrics=[utils.dice_coefficient])


callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best-custom-model.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]


history = model.fit(
    data.batched_train_dataset, 
    epochs=config.EPOCHS, 
    validation_data = data.batched_val_dataset
    )

#model.save('unet_model.h5') 

