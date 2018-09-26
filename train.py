
# coding: utf-8

# """
# 
# This notebook runs through Section 2.3: Style Transfer. 
# The images generated are similar to those in Figure 3. 
# 
# The Directory Paths block should be edited to suit local directory structure. 
# The Chosen Parameters block can be changed to try out different experimental settings. 
# 
# """

# In[ ]:

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim

import config as cfg
from utils.imaging import display_image, format_image, save_image
from utils.losses import gram_matrix, style_layer_loss
import vgg

# Set TF debugging to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Directories setup

if cfg.RESET_SAVES is True:
    # Ensure tensorboard is not running when you try to delete
    # this directory
    if os.path.exists(cfg.TENSORBOARD_DIR):
        shutil.rmtree(cfg.TENSORBOARD_DIR)
        
# Create the debug directory if it doesn't exist
# Tensorboard directory is made automatically if it doesn't exist
if os.path.exists(cfg.DEBUG_DIR):
    shutil.rmtree(cfg.DEBUG_DIR)
os.makedirs(cfg.DEBUG_DIR)

# Set the seeds to provide consistency between runs
# Can also comment out for variability between runs
np.random.seed(cfg.NP_SEED)
tf.set_random_seed(cfg.TF_SEED)


# The input node to the graph
input_var_initial_value = np.random.rand(
    1, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS)
input_var = tf.Variable(input_var_initial_value, dtype=tf.float32, name='input_var')

# Load the vgg model
with slim.arg_scope(vgg.vgg_arg_scope()):
    end_points = vgg.vgg_19_conv(input_var)

# Prepare to restore the vgg19 variables
# Skip trying to restore the input variable since it's new
all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(var_list=all_variables[1:])


# Load real images and create to-be-optimized image

# Construct the content image tensor
# And the graph operation which assigns it to input_var
content_image = plt.imread(cfg.CONTENT_IMAGE_PATH)
content_image = cv2.resize(content_image, (cfg.HEIGHT, cfg.WIDTH))
content_image_batch = np.expand_dims(content_image, axis=0)
content_image_batch = np.asarray(content_image_batch, dtype=np.float32)
content_image_tensor = tf.Variable(content_image_batch, dtype=tf.float32, name='content_image')

assign_content_image = tf.assign(input_var, content_image_tensor, name='assign_content_image')

# Construct the style image tensor
# And the graph operation which assigns it to input_var
style_image = plt.imread(cfg.STYLE_IMAGE_PATH)
style_image = cv2.resize(style_image, (cfg.HEIGHT, cfg.WIDTH))
style_image_batch = np.expand_dims(style_image, axis=0)
style_image_batch = np.asarray(style_image_batch, dtype=np.float32)
style_image_tensor = tf.Variable(style_image_batch, dtype=tf.float32, name='style_image')

assign_style_image = tf.assign(input_var, style_image_tensor, name='assign_style_image')

# Construct the white noise tensor
# And the graph operation which assigns it to input_var
white_noise = np.random.rand(cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS) * 255.
white_noise_batch = np.expand_dims(white_noise, axis=0)
white_noise_batch = np.asarray(white_noise_batch, dtype=np.float32)
white_noise_tensor = tf.Variable(white_noise_batch, dtype=tf.float32, name='white_noise')

assign_white_noise = tf.assign(input_var, white_noise_tensor, name='assign_white_noise')


# Set up content representation node
content_rep = end_points[cfg.CONTENT_LAYER]


# Obtain content representation for real image
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                  name='initialize_all')

with tf.Session() as sess:
    # Initialize new variables and then restore vgg_19 variables
    sess.run(init_op)
    saver.restore(sess, cfg.CHECKPOINT_PATH)
    
    assign_content_image.eval()
    
    real_content_rep_ = sess.run(content_rep)
    
real_content_rep = tf.constant(real_content_rep_, name='real_content_rep')


# Calculate the content loss
content_loss = tf.losses.mean_squared_error(labels=real_content_rep, predictions=content_rep)


# Set up gram matrix nodes
grams = []
filter_sizes = []
for i in range(cfg.CHOSEN_DEPTH):
    chosen_layer = end_points[cfg.STYLE_LIST[i]]
    gram_features = gram_matrix(chosen_layer)
    grams.append(gram_features)
    
    # Determine the size of the filters used at each layer
    # This is needed to calculate the loss from that layer
    _, filter_height, filter_width, _ = chosen_layer.get_shape().as_list()
    filter_size = float(filter_height * filter_width)
    filter_sizes.append(filter_size)    


# Obtain gram matrices for style image
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                  name='initialize_all')

with tf.Session() as sess:
    # Initialize new variables and then restore vgg_19 variables
    sess.run(init_op)
    saver.restore(sess, cfg.CHECKPOINT_PATH)
    
    assign_style_image.eval()
    
    real_image_grams = sess.run(grams)
    
# Create constants with the real image gram matrices
gram_constants = []
for i in range(cfg.CHOSEN_DEPTH):
    node_name = 'gram_constant_{}'.format(i)
    gram_constant = tf.constant(real_image_grams[i], name=node_name)
    gram_constants.append(gram_constant)


# Calculate the style loss
layer_losses = []
for i in range(cfg.CHOSEN_DEPTH):
    layer_loss = style_layer_loss(gram_constants[i], grams[i], filter_sizes[i])
    # Equal weighting on each loss, summing to 1
    layer_loss *= (1.0 / cfg.CHOSEN_DEPTH)
    layer_losses.append(layer_loss)
style_loss = tf.add_n(layer_losses, name='sum_layer_losses')


# Set up the final loss, optimizer, and summaries
loss = (cfg.CONTENT_WEIGHT * content_loss) + (cfg.STYLE_WEIGHT * style_loss)
optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE)
train_op = optimizer.minimize(loss, var_list=[input_var])

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                  name='initialize_all')

# Tensorboard summaries
loss_summary = tf.summary.scalar('loss', loss)
image_summary = tf.summary.image('image', input_var)


# Training
with tf.Session() as sess: 
    # Initialize all variables and then
    # restore weights for feature extractor
    sess.run(init_op)
    saver.restore(sess, cfg.CHECKPOINT_PATH)
    
    # Set up summary writer for tensorboard, saving graph as well
    train_writer = tf.summary.FileWriter(cfg.TENSORBOARD_DIR, sess.graph)
    
    # Assign the white noise to the variable being optimized
    assign_white_noise.eval()
    
    # Begin training
    for i in range(cfg.TRAINING_STEPS):
        loss_summary_, _ = sess.run([loss_summary, train_op])
        train_writer.add_summary(loss_summary_, i)
        
        if i % cfg.VALIDATION_STEPS == 0:
            image_summary_, current_image, loss_ = sess.run(
                [image_summary, input_var, loss])
            
            train_writer.add_summary(image_summary_, i)
            
            print('Step: {}, Loss: {}'.format(i, loss_))
            
            if cfg.DEBUG_OFFLINE is True:
                save_image(current_image, i, cfg.DEBUG_DIR)  
                
    # Display and save the final image
    current_image = input_var.eval()
    save_image(current_image, 'Final', cfg.DEBUG_DIR)

