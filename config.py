""" Data preprocessing settings"""
TRAIN_IMAGES_PATH = '/users/kate/downloads/train_v2'
TEST_IMAGES_PATH = '/users/kate/downloads/test_v2'
MASK_CSV_PATH = '/inputdata/train_ship_segmentations_v2.csv'
ORIGINAL_IMAGE_HEIGHT = 768
ORIGINAL_IMAGE_WIDTH = 768
TARGET_IMAGE_HEIGHT = 128
TARGET_IMAGE_WIDTH = 128

IMAGES_TOINCLUDE = 10
IMAGES_TOINCLUDE_TEST = 10

""" Data preparation settings"""
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 32

""" Model training settings"""
EPOCHS = 10
ADAM_LEARNING_RATE = 0.0001



""" PROCESSED DATA (if you have saved files from previous iterations)"""
images_file = '/inputdata/train_images.npy'
masks_file = '/inputdata/masks_images.npy'
test_images_file = '/inputdata/test_images.npy'
model_file ='/model/best_model.keras'