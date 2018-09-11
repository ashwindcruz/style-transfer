
# coding: utf-8

# """
# 
# This notebook runs through Section 2.2: Style Reconstructions. 
# The images generated are similar to those in Figure 1. 
# 
# The Directory Paths block should be edited to suit local directory structure. 
# The Chosen Parameters block can be changed to try out different experimental settings. 
# 
# """

# In[1]:

import cv2
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.imaging import display_image, format_image, save_image
import vgg


# ### Directory paths

# In[2]:

# Path to image we are extracting content from
real_image_path = './starry_night.jpg'

# Path to vgg19 checkpoint, must be downloaded separately
checkpoint_path = './vgg_19.ckpt'

# Location of tensorboard summaries
tensorboard_dir = './train/'

# Path to directory used for storing images
debug_dir = './debug/'

# Determines whether information is saved between runs
# for tensorboard
reset_saves = True
if reset_saves is True:
    # Ensure tensorboard is not running when you try to delete
    # this directory
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
        
# Create the debug directory if it doesn't exist
# Tensorboard directory is made automatically if it doesn't exist
# if os.path.exists(debug_dir):
#     shutil.rmtree(debug_dir)
# os.makedirs(debug_dir)


# ### Chosen parameters

# In[3]:

dimensions_list = [512]
depth_list = [3]
seed_list = [0,1,2]

for m in range(len(dimensions_list)):
    # Dimensions desired for image, channels must be kept as 3
    height = dimensions_list[m]
    width = dimensions_list[m]
    channels = 3

    for j in range(len(seed_list)):
        # Set the seeds to provide consistency between runs
        # Can also comment out for variability between runs
        np.random.seed(seed_list[j])
        tf.set_random_seed(seed_list[j])

        # Layer being used to produce features
        feature_layer_list = ['vgg_19/conv1/conv1_1', 'vgg_19/conv2/conv2_1', 'vgg_19/conv3/conv3_1',
            'vgg_19/conv4/conv4_1', 'vgg_19/conv5/conv5_1']

        for k in range(len(depth_list)):

            tf.reset_default_graph()

    

            # Chosen depth corresponds to how many feature layers you want to use
            chosen_depth = depth_list[k]

            # Learning rate for optimizer
            learning_rate = 1e-1

            # Number of training and validation step
            # In this instance, validation refers to when we would like to examine the 
            # currently optimized image, save it, and loss
            training_steps = 100000
            validation_steps = 10000

            # Online debugging refers to images that will be displayed within the notebook 
            # using plt
            # Offline debugging refers to images that will be saved to folder using plt
            debug_online = True
            debug_offline = True


            # ### Function to compute the Gram matrix

            # In[4]:

            def gram_matrix(feature_set):
                """
                Given a set of vectors, in the form of a tensor, from a layer, 
                compute the Gram matrix (https://en.wikipedia.org/wiki/Gramian_matrix).
                
                Args:
                    feature_set: Tensor of vectors 
                        ([1, filter_height, filter_width, num_feature_maps])
                Returns:
                    gram_matrix: Computed Gram matrix ([num_feature_maps, num_feature_maps])
                """
                
                batch_size, filter_height, filter_width, num_feature_maps =         feature_set.get_shape().as_list()
                feature_set = tf.reshape(
                    feature_set, [filter_height * filter_width, num_feature_maps], name='vectorize_map')
                gram_matrix = tf.matmul(
                    feature_set, feature_set, transpose_a=True, name='gram_map')
                
                return gram_matrix


            # ### Function to compute the style loss of a single layer

            # In[5]:

            def style_layer_loss(gram_matrix_desired, gram_matrix_predicted, filter_size):
                """
                Compute the loss between the gram matrix of the styling image and the
                gram matrix of the image undergoing optimization. 
                
                Args:
                    gram_matrix_desired  : Gram matrix of the styling image
                    gram_matrix_predicted: Gram matrix of the image undergoing optimization. 
                    filter_size          : The size of an individual filter map (filter_height * filter_width)
                Returns: 
                    loss_contribution: The loss contribution from this layer 
                """
                
                num_filters, _ = gram_matrix_desired.get_shape().as_list()
                num_filters = float(num_filters)
                summed_squared_difference = tf.reduce_sum(
                    tf.square(gram_matrix_predicted - gram_matrix_desired), name='summed_squared_diff')
                loss_contribution = (1 / (4 * np.power(num_filters, 2) * np.power(filter_size, 2)))         * summed_squared_difference
                    
                return loss_contribution


            # ### Set up input node and feature extractor

            # In[6]:

            # The input node to the graph
            # These values are what is required by vgg19 for height, width, channels
            input_var_initial_value = np.random.rand(1, height, width, channels)
            input_var = tf.Variable(input_var_initial_value, dtype=tf.float32, name='input_var')

            # Load the vgg model
            with slim.arg_scope(vgg.vgg_arg_scope()):
                end_points = vgg.vgg_19_conv(input_var)


            # ### Set up restoring feature extractor weights

            # In[7]:

            # Prepare to restore the vgg19 nodes
            # Skip trying to restore the input variable since it's new
            all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(var_list=all_variables[1:])


            # ### Load real image and create to-be optimised image

            # In[8]:

            # Construct the real image tensor
            # And the graph operation which assigns it to input_var
            real_image = plt.imread(real_image_path)
            real_image = cv2.resize(real_image, (height, width))
            real_image_batch = np.expand_dims(real_image, axis=0)
            real_image_batch = np.asarray(real_image_batch, dtype=np.float32)
            real_image_tensor = tf.Variable(real_image_batch, dtype=tf.float32, name='real_image')

            assign_real_image = tf.assign(input_var, real_image_tensor, name='assign_real_image')

            # Construct the white noise tensor
            # And the graph operation which assigns it to input_var
            white_noise = np.random.rand(height, width, channels) * 255.
            white_noise_batch = np.expand_dims(white_noise, axis=0)
            white_noise_batch = np.asarray(white_noise_batch, dtype=np.float32)
            white_noise_tensor = tf.Variable(white_noise_batch, dtype=tf.float32, name='white_noise')

            assign_white_noise = tf.assign(input_var, white_noise_tensor, name='assign_white_noise')


            # ### Set up gram matrix nodes

            # In[9]:

            # Choose which layers will be used to 
            # reconstruct the original image
            grams = []
            filter_sizes = []
            for i in range(chosen_depth):
                chosen_layer = end_points[feature_layer_list[i]]
                gram_features = gram_matrix(chosen_layer)
                grams.append(gram_features)
                
                # Determine the size of the filters used at each layer
                # This is needed to calculate the loss from that layer
                _, filter_height, filter_width, _ = chosen_layer.get_shape().as_list()
                filter_size = float(filter_height * filter_width)
                filter_sizes.append(filter_size)    


            # ### Obtain gram matrices for real image

            # In[10]:

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                              name='initialize_all')

            with tf.Session() as sess:
                # Initialize new variables and then restore vgg_19 variables
                sess.run(init_op)
                saver.restore(sess, checkpoint_path)
                
                assign_real_image.eval()
                
                real_image_grams = sess.run(grams)
                
            # Create constants with the real image gram matrices
            gram_constants = []
            for i in range(chosen_depth):
                node_name = 'gram_constant_{}'.format(i)
                gram_constant = tf.constant(real_image_grams[i], name=node_name)
                gram_constants.append(gram_constant)


            # ### Calculate the loss for each layer

            # In[11]:

            layer_losses = []
            for i in range(chosen_depth):
                layer_loss = style_layer_loss(gram_constants[i], grams[i], filter_sizes[i])
                # Equal weighting on each loss, summing to 1
                layer_loss *= (1.0 / chosen_depth)
                layer_losses.append(layer_loss)


            # ### Set up the loss, optimizer, and summaries

            # In[12]:

            # Set up final loss term and optimizer
            loss = tf.add_n(layer_losses, name='sum_layer_losses')
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, var_list=[input_var])

            # Initializers
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                              name='initialize_all')

            # Tensorboard summaries
            loss_summary = tf.summary.scalar('loss', loss)
            image_summary = tf.summary.image('image', input_var)
            merged_summary = tf.summary.merge_all()


            # ### Training

            # In[13]:

            with tf.Session() as sess: 
                # Initialize all variables and then
                # restore weights for feature extractor
                sess.run(init_op)
                saver.restore(sess, checkpoint_path)
                
                # Set up summary writer for tensorboard, saving graph as well
                train_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
                
                # Assign the white noise to the variable being optimized
                assign_white_noise.eval()
                
                # Begin training
                for i in range(training_steps):
                    loss_summary_, _ = sess.run([loss_summary, train_op])
                    train_writer.add_summary(loss_summary_, i)
                    
                    if i % validation_steps == 0:
                        image_summary_, current_image, loss_ = sess.run(
                            [image_summary, input_var, loss])
                        
                        train_writer.add_summary(image_summary_, i)
                        
                        print('Step: {}, Loss: {}'.format(i, loss_))
                        
                        # if debug_online is True:
                        #     display_image(current_image, i)
                        # if debug_offline is True:
                        #     save_image(current_image, i, debug_dir)  
                            
                # Display and save the final image
                current_image = input_var.eval()
                #display_image(current_image, 'Final')
                save_image(current_image, 'Final', debug_dir)

                current_image_name = '{}_{}_{}.png'.format(dimensions_list[m],depth_list[k],seed_list[j])
                save_image(current_image, current_image_name, './script_saved')

