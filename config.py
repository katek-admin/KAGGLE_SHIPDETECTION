""" Data preparation settings"""
TRAIN_IMAGES_PATH = '/Users/kate/Downloads/train_v2'
TEST_IMAGES_PATH = '/Users/kate/Downloads/test_v2'
MASK_CSV_PATH = '/inputdata/train_ship_segmentations_v2.csv'
ORIGINAL_IMAGE_HEIGHT = 768
ORIGINAL_IMAGE_WIDTH = 768
TARGET_IMAGE_HEIGHT = 128
TARGET_IMAGE_WIDTH = 128

IMAGES_TOINCLUDE = 3000
IMAGES_TOINCLUDE_TEST = 30

""" Model training settings"""
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

""" TRAINING DATA (TEMP)"""
images_file = '/inputdata/train_images.npy'
masks_file = '/inputdata/masks_images.npy'
test_images_file = '/inputdata/test_images.npy'
model_file ='/model/checkpoint-23.keras'