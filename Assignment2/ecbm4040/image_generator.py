#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to False.
        #
        # Hint: Since you may directly perform transformations on x and y, and don't want your original data to be contaminated 
        # by those transformations, you should use numpy array build-in copy() method. 
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        self.x = x
        self.y = y
        self.num_of_samples = x.shape[0]



        # One way to use augmented data is to store them after transformation (and then combine all of them to form new data set).
        # Following variables (along with create_aug_data() function) is one kind of implementation.
        # You can either figure out how to use them or find out your own ways to create the augmented dataset.
        # if you have your own idea of creating augmented dataset, just feel free to comment any codes you don't need


    def create_aug_data(self):
        # If you want to use function create_aug_data() to generate new dataset, you can perform the following operations in each
        # transformation function:
        #
        # 1.store the transformed data with their labels in a tuple called self.translated, self.rotated, self.flipped, etc. 
        # 2.increase self.N_aug by the number of transformed data,
        # 3.you should also return the transformed data in order to show them in task4 notebook
        # 
        
        '''
        Combine all the data to form a augmented dataset 
        '''
        
        #set x and y as empty
        self.x_aug = None
        self.y_aug = None
        
        # set the length of the regular and augmented datasets
        N = [self.x.shape[0],self.x.shape[0],self.x.shape[0],self.x.shape[0],int(self.x.shape[0]/10)]
        
        #set augmented x and y to regular x and y
        self.x_aug = self.x
        self.y_aug = self.y
        
        # set values for trans, rot, flip, and noise
        (shift_h,shift_w) = np.random.random_integers(-5,5,2)
        ang = np.random.uniform(-180,180)
        flip = np.random.choice(['h','v','hv'])
        amp = np.random.uniform(0,.05)
        portion = .1

        # translate images and add them to x and y, then remove array
        translated = self.translate(shift_h,shift_w)
        self.x_aug = np.vstack((self.x_aug,translated[0]))
        self.y_aug = np.hstack((self.y_aug,translated[1]))
        translated = None
        
        # rotat images and add them to x and y, then remove array
        rotated = self.rotate(ang)
        self.x_aug = np.vstack((self.x_aug,rotated[0]))
        self.y_aug = np.hstack((self.y_aug,rotated[1]))
        rotated = None
        
        # flip images and add them to x and y, then remove array
        flipped = self.flip(flip)
        self.x_aug = np.vstack((self.x_aug,flipped[0]))
        self.y_aug = np.hstack((self.y_aug,flipped[1]))
        flipped = None
        
        # add noise to images and add them to x and y, then remove array
        added = self.add_noise(portion,amp)
        self.x_aug = np.vstack((self.x_aug,added[0]))
        self.y_aug = np.hstack((self.y_aug,added[1]))
        added = None
        
        # return augmented dataset with labels
        return (self.x_aug,self.y_aug)

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        #######################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        
        # set total possible and count
        tot_poss = int(self.num_of_samples/batch_size) - 1
        count = 0
        
        # eternal loop
        while True:
            # if the count is less than the total possible
            if (count < tot_poss):
                # incriment and return a batch
                count += 1
                yield((self.x_aug[count*batch_size:(count+1)*batch_size],self.y_aug[count*batch_size:(count+1)*batch_size]))
            else:
                
                # if whole dataset has been iterated through, create a new augmented dataset
                _ = self.create_aug_data()
                
                # if shuffle is true, shuffle data
                if shuffle:
                    s = np.arange(self.x_aug.shape[0])
                    shuff = np.random.shuffle(s)
                    self.x_aug = self.x_aug[s]
                    self.y_aug = self.y_aug[s]
                
                # reset counter
                count = 0
                
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        # show 4x4 of top images in images
        fig, axes1 = plt.subplots(4,4,figsize=(8,8))
        i = 0
        for j in range(4):
            for k in range(4):
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(images[i:i+1][0])
                i += 1
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """

        # TODO: Implement the translate function. Remember to record the value of the number of pixels shifted.
        # Note: You may wonder what values to append to the edge after the shift. Here, use rolling instead. For
        # example, if you shift 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        # Hint: Numpy.roll (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        
        # roll the image by height and width
        translated = (np.roll(self.x,shift = (shift_height,shift_width), axis = (1,2)),self.y)
        return translated
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        # TODO: Implement the rotate function. Remember to record the value of
        # rotation degree.
        
        # rotate the image and return
        rotated = (rotate(self.x,angle,axes =(1,2),reshape=False),self.y)
        return rotated
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        # TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
        # is_vertical_flip.
        
        # flip based on mode
        if mode == 'h':
            flipped = (np.flip(self.x, axis=1),self.y)
        elif mode == 'v':
            flipped = (np.flip(self.x, axis=2),self.y)
        else:
            flipped = np.flip(self.x,axis=1)
            flipped = (np.flip(flipped,axis=2),self.y)
            
        return flipped
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        # TODO: Implement the add_noise function. Remember to record the
        # boolean value is_add_noise. You can try uniform noise or Gaussian
        # noise or others ones that you think appropriate.
        
        # create a list of the index values
        index = np.linspace(0,self.x.shape[0]-1,num=self.x.shape[0],dtype=int)
        # pick out portion % of the indexes randomly
        p = np.random.choice(index,size=int(self.x.shape[0]*portion),replace=False)
        # create the portion of the x that noise will be added to
        portion = self.x[p,:,:,:]
        # add noise of amplitude to the portion
        portion += np.random.normal(0,amplitude,portion.shape)
        
        added =(portion,self.y[p])
            
        return added
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
