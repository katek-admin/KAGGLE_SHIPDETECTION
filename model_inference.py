import tensorflow as tf
import os

import data, config, utils
from unet_model import model


def show_predictions(model, dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    if not dataset:
        data.load_test_data()

    if isinstance(dataset.element_spec, tf.TensorSpec):
        for image in dataset.take(num):
            pred_mask = model.predict(image)
            utils.display([image[0], utils.create_mask(pred_mask)])
    else:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            utils.display([image[0], mask[0], utils.create_mask(pred_mask)])
   
        
# Load the model from the saved file
if os.path.exists(os.getcwd()+ config.model_file):
    model = tf.keras.models.load_model(os.getcwd()+ config.model_file, custom_objects={'combined_loss': utils.combined_loss, 'dice_coefficient': utils.dice_coefficient})
else:
    model = model

#dataset = data.load_test_data()
dataset = data.batched_train_dataset
show_predictions(model = model, dataset = dataset, num = 10)

