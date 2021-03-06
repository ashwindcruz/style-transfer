# Path to images we are extracting content and style from
CONTENT_IMAGE_PATH = './coastal_scene.jpg'
STYLE_IMAGE_PATH = './starry_night.jpg'

# Seed for initializing numpy and tf
NP_SEED = 0
TF_SEED = 0

# Path to vgg19 checkpoint, must be downloaded separately
CHECKPOINT_PATH = './vgg_19.ckpt'

# Location of tensorboard summaries
TENSORBOARD_DIR = './train/'

# Path to directory used for storing images
DEBUG_DIR = './debug/'

# Dimensions desired for image, channels must be kept as 3
HEIGHT = 224
WIDTH = 224
CHANNELS = 3

# Layer being used for content component
CONTENT_LAYER = 'vgg_19/conv2/conv2_2'

# Layers that can be used for style component
STYLE_LIST = [
'vgg_19/conv1/conv1_1',
'vgg_19/conv2/conv2_1', 
'vgg_19/conv3/conv3_1',
'vgg_19/conv4/conv4_1', 
'vgg_19/conv5/conv5_1'
]

# Chosen depth corresponds to how many feature layers you want to use
# for the style component
CHOSEN_DEPTH = 2

# Weights for each loss component
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = CONTENT_WEIGHT/8e-4

# Learning rate for optimizer
LEARNING_RATE = 1e-1

# Number of training and validation step
# In this instance, validation refers to when we would like to examine:
# save currently optimized image and loss
TRAINING_STEPS = 10000
VALIDATION_STEPS = 1000

# Offline debugging refers to images that will be saved to folder using plt, 
# every validation step
DEBUG_OFFLINE = True

# Determines whether information is saved between runs
# for tensorboard
RESET_SAVES = True