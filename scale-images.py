# Rescale all images to (32*32) size to make the shapes consistent
from skimage.transform import resize
from skimage import io
import numpy as np
import os

directory = 'images'

images = os.listdir(directory)

for image_name in images:
	image_data = io.imread(directory + '/' + image_name)

	# Pad the image
	width, height = image_data.shape

	# Create an empty square array
	padded_image = np.zeros((max(image_data.shape), max(image_data.shape)))

	# Find the image center
	center_width = (max(image_data.shape) - width) // 2
	center_height = (max(image_data.shape) - height) // 2

	# Insert the original image data in the padded image
	padded_image[center_width: center_width + width, center_height: center_height + height] = image_data

	# Scale the image
	scaled_image = resize(np.uint8(padded_image), (32, 32))

	io.imsave('processed/{}'.format(image_name), scaled_image)
