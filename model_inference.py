import matplotlib.pyplot as plt
import numpy as np
import os

from keras import models

import data
from unet_model_v3 import dice_coefficient, model

#test_images = data.load_test_data()

# Load the model from the saved file
if os.path.exists('unet_model.h5'):
    model = models.load_model('unet_model.h5')

# Predict and visualize
preds = model.predict(data.images)
# Apply threshold to get binary mask
binary_predictions = (preds > 0.5).astype(np.uint8)

for i in range(10):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(data.images[i])
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(data.masks[i], cmap='gray')
    plt.title("True Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(binary_predictions[i], cmap='gray')
    plt.title("Predicted Mask")

    print(dice_coefficient(data.masks[i], preds[i]))
    #print(dice_coefficient(data.masks[i], binary_predictions[i]))
    plt.show()