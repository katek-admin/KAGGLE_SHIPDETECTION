import tensorflow as tf
import config
import data
from unet_model import model

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

model.save('unet_model.h5') 

