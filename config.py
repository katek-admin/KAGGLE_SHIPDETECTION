""" Data preparation settings"""
IMAGES_TOINCLUDE = 1000
TRAIN_IMAGES_PATH = '/Users/kate/Downloads/train_v2'
TEST_IMAGES_PATH = '/Users/kate/Downloads/test_v2'
MASK_CSV_PATH = '/inputdata/train_ship_segmentations_v2.csv'
ORIGINAL_IMAGE_HEIGHT = 768
ORIGINAL_IMAGE_WIDTH = 768
TARGET_IMAGE_HEIGHT = 256
TARGET_IMAGE_WIDTH = 256
#input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

""" Model training settings"""
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1