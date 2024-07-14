import tensorflow as tf
from keras import optimizers
import os

import config, data, utils


def train():
    from unet_model import model

    # Load the model from the saved file
    if os.path.exists(os.getcwd()+ config.model_file):
        print('Saved model is taken')
        model = tf.keras.models.load_model(os.getcwd()+ config.model_file, custom_objects={'combined_loss': utils.combined_loss, 'dice_coefficient': utils.dice_coefficient})
    else:
       
        model.compile(optimizer=optimizers.Adam(learning_rate=config.ADAM_LEARNING_RATE, epsilon=1e-06), loss=[utils.combined_loss], metrics=[utils.dice_coefficient])

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.getcwd() + config.model_file,
                monitor="val_loss",
                save_best_only=True,
            )
        ]

        print("Model fit starting...")
        history = model.fit(
            data.batched_train_dataset, 
            epochs=config.EPOCHS, 
            validation_data = data.batched_val_dataset,
            callbacks=[callbacks_list]
            )
    
    return model

trained_model = train()

