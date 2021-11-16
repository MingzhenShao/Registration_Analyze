import tensorflow as tf 
import numpy as np 
import os, os.path
import cv2, re, random, csv
# import oct, deformation
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
from glob import glob
from numpy.linalg import inv

import scipy.ndimage

from scipy import stats
from multiprocessing import Process, Queue
import datetime

#################
#		In the deformation task, we use oct.ConvertH2CBGR2Gray
#		This norm the value in [0,1], While outhrs with the range in [0,255]
#
##################

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

run_id = 'baseline'
log_dir = './H_predict/log/{}'.format(run_id)
ckpt_dir = './H_predict/ckpt/{}'.format(run_id)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
import logging
'''
logging.basicConfig(filename=log_dir + '/H_predict_val.log',
														filemode='a', level=logging.DEBUG,
														format='%(asctime)s, %(msecs)d %(message)s',
														datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

#############################
logging.info('\n\nH_predict...\n') 
'''
######## Pre-method ############

def homography_regression_model(input_img):
	
	x = tf.layers.conv2d(inputs=input_img, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)

	
	
	# without fc layer (dense)	
	# It's hard to analyze how fc layers works.
	
	x = tf.layers.conv2d(inputs=x, filters=8, strides=1, kernel_size=3, padding="same", activation=None)
	x1 = x
	out = tf.reduce_mean(x, axis=[1,2])
	'''

	x = tf.layers.flatten(x)
	#x = tf.layers.dropout(x, rate=0.75)
	x = tf.layers.dense(x, 1024)
	x1 = x
	out = tf.layers.dense(x, 8)
	'''
	return out, x1


# # function for training and test
# def get_train(path = "~/Documents/DATASET/VOC2011/train/*.jpg", num_examples = 1280):
# 	# hyperparameters
# 	rho = 32
# 	patch_size = 224
# 	height = 320
# 	width = 320

# 	loc_list = glob(path)
# 	X = np.zeros((num_examples,128, 128, 2))  # images
# 	Y = np.zeros((num_examples,8))
# 	for i in range(num_examples):
# 		# select random image from tiny training set
# 		index = random.randint(0, len(loc_list)-1)
# 		img_file_location = loc_list[index]
# 		color_image = plt.imread(img_file_location)
# 		color_image = cv2.resize(color_image, (width, height))
# 		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

# 		# create random point P within appropriate bounds
# 		y = random.randint(rho, height - rho - patch_size)  # row?
# 		x = random.randint(rho, width - rho - patch_size)  # col?
# 		# define corners of image patch
# 		top_left_point = (x, y)
# 		bottom_left_point = (patch_size + x, y)
# 		bottom_right_point = (patch_size + x, patch_size + y)
# 		top_right_point = (x, patch_size + y)
# 		four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
# 		perturbed_four_points = []
# 		for point in four_points:
# 			perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

# 		# compute H
# 		H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
# 		H_inverse = inv(H)
# 		inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
# 		warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

# 		# grab image patches
# 		original_patch = gray_image[y:y + patch_size, x:x + patch_size]
# 		warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
# 		# make into dataset
# 		training_image = np.dstack((original_patch, warped_patch))
# 		H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
# 		X[i, :, :] = training_image
# 		Y[i, :] = H_four_points.reshape(-1)		
# 	return X,Y


def pre_deform(path, num_examples = 256):
	loc_list = glob(path)
	X = []
	Y = []
	for i in range(num_examples):
		index = random.randint(0, len(loc_list)-1)
		img_file_location = loc_list[index]
		color_image = cv2.imread(img_file_location)
			
		try:
			gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
		except Exception as e:
			print(img_file_location)
			continue
		ori, deform = deformation.Deform(gray_image, 10, img_size=512)
		X.append(ori)
		Y.append(deform)
	return X, Y

def get_generator(queue, path, num_examples = 256):

	for k in range(1000):
		# ori_data, deform_data = pre_deform(path, num_examples*4)
		for k in range(120):
	#	while 1:
			# hyperparameters
			rho = 32
			patch_size = 224
			height = 320
			width = 320

			loc_list = glob(path)

			X = np.zeros((num_examples,224, 224, 2))  # images
			Y = np.zeros((num_examples,8))
			for i in range(num_examples):
				# select random image from tiny training set
				index = random.randint(0, len(loc_list)-1)

				# index = random.randint(0, len(ori_data)-1)
				# ori = ori_data[index]
				# deform = deform_data[index]
				# ori = cv2.resize(ori, (width, height))
				# deform = cv2.resize(deform, (width, height))

				#### White Noise Image ###
				# img_data = []
				# pixel_color = ""
				
				# gray_image = Image.new("L", size=(320, 320))
				# for k in range(320*320):
				# 	img_data.append(random.randint(0, 255))
				# gray_image.putdata(img_data)
				# gray_image = np.array(gray_image)

				############################

				img_file_location = loc_list[index]
				color_image = cv2.imread(img_file_location)
				
				try:
					gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
				except Exception as e:
					print(img_file_location)
					continue
				gray_image = cv2.resize(gray_image, (width, height))
				
				# create random point P within appropriate bounds
				y = random.randint(rho, height - rho - patch_size)  # row
				x = random.randint(rho, width - rho - patch_size)  # col
				# define corners of image patch
				top_left_point = (x, y)
				bottom_left_point = (patch_size + x, y)
				bottom_right_point = (patch_size + x, patch_size + y)
				top_right_point = (x, patch_size + y)
				four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
				perturbed_four_points = []
				for point in four_points:
					perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

				# compute H
				H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
				H_inverse = inv(H)
				inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 320))
				
				# grab image patches
				original_patch = gray_image[y:y + patch_size, x:x + patch_size]
				warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]

				# make into dataset
				training_image = np.dstack((original_patch, warped_patch))
				H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))


				X[i, :, :] = training_image
				Y[i, :] = H_four_points.reshape(-1)		
			queue.put((np.array(X), np.array(Y)), block = True)
	#	yield (X,Y)
	return

def get_test(path):

	rho = 32
	patch_size = 224
	height = 320
	width = 320
	# #random read image
	# loc_list = glob(path)
	# index = random.randint(0, len(loc_list)-1)
	# img_file_location = loc_list[index]

	# #For *.png
	# if(img_file_location.split('.')[-1] == 'png'):
	# 	color_image = np.array(cv2.imread(img_file_location))		# Why use np.array(Image.open(img_file_location)) for *.png ?
	# else:
	# color_image = cv2.imread(img_file_location)
	color_image = cv2.imread(path)

	try:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(path)
		# print(img_file_location)
	
	gray_image = cv2.resize(gray_image, (width, height))
	#points
	######deformation image#######
	# ori, deform = deformation.Deform(gray_image, 10, img_size=512)
	# ori = cv2.resize(ori, (width, height))
	# deform = cv2.resize(deform, (width, height))
	# ori = np.float32(ori)/255
	# deform = np.float32(deform)/255

	### White Noise Image ###

	# img_data = []
	# gray_image = Image.new("L", size=(320, 320))
	# for i in range(320*320):
	# 	img_data.append(random.randint(0, 255))
	# gray_image.putdata(img_data)
	# gray_image = np.array(gray_image)


	y = random.randint(rho, height - rho - patch_size)  # row
	x = random.randint(rho,  width - rho - patch_size)  # col
	top_left_point = (x, y)
	bottom_left_point = (patch_size + x, y)
	bottom_right_point = (patch_size + x, patch_size + y)
	top_right_point = (x, patch_size + y)
	four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
	four_points_array = np.array(four_points)
	perturbed_four_points = []
	for point in four_points:
		perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
		
	#compute H
	H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	H_inverse = inv(H)
	inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	
	# make into dataset
	training_image = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))
	H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
	
	# print(index, x, y, random.randint(-rho, rho), random.randint(-rho, rho))

	return training_image, H_four_points.reshape(-1), np.array(four_points).reshape(-1), color_image, gray_image, inv_warped_image
	

def get_test_demo(path, rand_list):
	rho = 32
	patch_size = 224
	height = 320
	width = 320
	#random read image
	loc_list = glob(path)
	index = rand_list[0]
	img_file_location = loc_list[index]

	#For *.png
	if(img_file_location.split('.')[-1] == 'png'):
		color_image = np.array(Image.open(img_file_location))
	else:
		color_image = cv2.imread(img_file_location)

	try:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(img_file_location)

	gray_image = cv2.resize(gray_image,(width,height))

	#points
	y = rand_list[1]  # row
	x = rand_list[2]  # col
	top_left_point = (x, y)
	bottom_left_point = (patch_size + x, y)
	bottom_right_point = (patch_size + x, patch_size + y)
	top_right_point = (x, patch_size + y)
	four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
	four_points_array = np.array(four_points)
	perturbed_four_points = []
	for point in four_points:
		perturbed_four_points.append((point[0] + rand_list[3], point[1] + rand_list[4]))
		
	#compute H
	H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	H_inverse = inv(H)
	inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	# make into dataset
	training_image = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))
	H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
	
	return training_image, H_four_points.reshape(-1), np.array(four_points).reshape(-1), color_image
	
	# return color_image, H_inverse,val_image,four_points_array

def get_test_visualization(loc_list, index, H_four_points, four_points):

	H_four_points = H_four_points.reshape([4,2])
	four_points = four_points.reshape([4,2])
	rho = 32
	patch_size = 224
	height = 320
	width = 320
	# #random read image
	# loc_list = glob(path)
	# index = random.randint(0, len(loc_list)-1)
	img_file_location = loc_list[index]

	try:
		color_image = cv2.imread(img_file_location)
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(img_file_location)
	
	gray_image = cv2.resize(gray_image, (width, height))

	y = four_points[0][0]   # row
	x = four_points[0][1] # col
	
	perturbed_four_points = four_points + H_four_points
	
	#compute H
	#print(four_points, perturbed_four_points)
	H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	H_inverse = inv(H)
	inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	
	# make into dataset
	val_images = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))

	# print(index, x, y, random.randint(-rho, rho), random.randint(-rho, rho))

	return val_images, gray_image, inv_warped_image, original_patch, warped_patch

def consumer(queue):
	X, Y = queue.get(block = True)
	return X, Y


epochs = 1000
batch_size = 64
data_batch = np.zeros((batch_size,256,256,1))
labels_batch = np.zeros((batch_size, 8))

#logging.info("Loading Train Data...")

#g = get_generator(path = "./eye/*.png", num_examples=batch_size, eye=0)

# logging.info( "Loading Val Data...")
# data_V, label_V = data_loader('./val.txt')
# data_V = np.asarray(data_V)
# label_V = np.asarray(label_V)


def result_show_diff(name, gray_image, inv_warped_image, pred_val, H_points, base_points):		# This method leads to a resolution decrease. 

	print(name, pred_val, H_points, base_points)
	#quit()
	
	perturbed_four_points = base_points + pred_val 		#use H_points to show it's right
	H = cv2.getPerspectiveTransform(np.float32(perturbed_four_points.reshape([4,2])), np.float32(base_points.reshape([4,2])))
	H_inverse = inv(H)
	predict_image = cv2.warpPerspective(inv_warped_image, H_inverse, (320, 320))
	diff_img = predict_image - gray_image

	plt.imshow(predict_image)
	plt.savefig('{}pred.png'.format(name))
	plt.imshow(gray_image)
	plt.savefig('{}gray.png'.format(name))
	plt.imshow(diff_img)
	plt.colorbar()
	plt.savefig('{}diff.png'.format(name))
	
	plt.close('all')



def result_show_rectangle(gray_image, inv_warped_image, original_patch, warped_patch, pred_val, H_points, base_points):
	pass

	transform_img = cv2.warpPerspective(cv2.resize(ori_img,(320,320)), H_inverse, (320, 320))
	# transform_img = deform_img 
	transform_img_ = transform_img.copy()
	color_image = ori_img 

	ax=plt.subplot(2,3,1)
	# plt.title("ori_img")
	color_image_draw = cv2.polylines(color_image, np.array([base_points.reshape((4,2))], np.int32), 1, (255,0,0), 2)
	ax.imshow(color_image_draw, vmin=0, vmax=255, cmap='jet')

	
	label_4point = (-H_points + base_points).reshape((4,2))	
	pred_4point = np.int32(np.reshape((-pred + base_points),(4,2)))
	# print(pred_4point, label_4point)

	cv2.polylines(transform_img, np.array([label_4point], np.int32), 1, (255,0,0), 2)
	# cv2.polylines(transform_img, np.array([pred_4point], np.int32), 1, (0,0,255), 2)
	# cv2.polylines(transform_img, np.array([base_points.reshape((4,2))], np.int32), 1, (0,0,255), 2)
	# mce = np.sum(abs(pred-H_points))/8
	# print(mce)

	# cv2.putText(transform_img,'Mean Corner Error = {0:.2f}'.format(mce),(20,20), 4, 0.5, (255,0,0),1,cv2.LINE_AA)

	ax=plt.subplot(2,3,2)
	# plt.title("transform_img")
	ax.imshow(transform_img, vmin=0, vmax=255, cmap='jet')

	ax=plt.subplot(2,3,3)

	cv2.polylines(transform_img_, np.array([base_points.reshape((4,2))], np.int32), 1, (0,0,255), 2)
	# plt.title("transform_img")
	ax.imshow(transform_img_, vmin=0, vmax=255, cmap='jet')

	# plt.subplots_adjust(wspace=-0.35)
	plt.savefig('{}.png'.format(item))
	
	plt.show()

def batch_norm_featuremap(pred_val, feature_map):
	pass
	pred_val_idx = []
	feature_map_idx = []
	pred_val_idx.append(pred_val)
	feature_map_idx.append(feature_map)

	max_feature = np.amax(feature_map_idx)
	min_feature = np.min(feature_map_idx)
	feature_range = max_feature - min_feature

	norm_feature = (feature_map-min_feature)/feature_range



def save_fig(pred_val, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, feature_map, H_points, base_points, boardline_feature, feature_max=None, feature_min=None):

	name = 'S_' + str(np.int(np.mean(np.abs(pred_val - H_points)))) + '_S_'

	for e in pred_val:
		name += str(e)
		name += '_'
	# sub_img, ax = plt.subplots(2,2, tight_layout=True)

	# # Show image pairs	
	# for e in ax.reshape(-1):
	# 	e.axis('off')
	# ax[0,0].imshow(gray_image, cmap='gray')
	# ax[0,1].imshow(inv_warped_image, cmap='gray')
	# ax[1,0].imshow(original_patch, cmap='gray')
	# ax[1,1].imshow(warped_patch, cmap='gray')
	# plt.savefig(save_dir+'/{}img.png'.format(name))
	# plt.close('all')

	# Show feature maps
	
	# fig_f, ax_f = plt.subplots(2,4, tight_layout=True)
	# idx = 0
	# for e in ax_f.reshape(-1):
	# 	e.axis('off')
	# 	e.imshow(feature_map[0,:,:,idx], vmin=feature_min, vmax=feature_max, cmap='gray')
	# 	idx += 1

	# # colorbar() ?!

	# plt.savefig(save_dir+'/{}feature.png'.format(name))
	# plt.close('all')


	# Give the hist of each channel
	
	_feature_map = np.around(np.array(feature_map), 2)

	# fig_m, ax_m = plt.subplots(2,4, tight_layout=True, figsize=(8,4))
	# idx = 0
	# for e in ax_m.reshape(-1):
	# 	medium = np.sort(_feature_map[0,:,:,idx], axis=None)
	# 	e.hist(medium)
	# 	idx += 1
	# plt.savefig(save_dir+'/{}matrix.png'.format(name))
	# plt.close('all')

	###################################
	# Show focus on the input image for each channel
	###################################

	fig_F, ax_F = plt.subplots(2,4, sharex='all', sharey='all', tight_layout=True, figsize=(12,6))			
	idx = 0
	
	sub_ave_list = []
	for e in ax_F.reshape(-1):
		if(abs(pred_val[idx]-H_points[idx])>5):				# if the error is too large, do NOT visualize this feature.
			# bin_feature_map = None
			e.text(0.2, 0.2, 'Loss error', ha='center', va='center', fontsize=28, color='C1')
			# idx += 1
			# print("skip!")
			# continue
		bin_feature_map = _feature_map[0,:,:,idx].copy()

		sub_feature_map = bin_feature_map - boardline_feature[0,:,:,idx]
		sub_feature_map = np.around(sub_feature_map, 0)

		_sub_feature_map = np.zeros(np.shape(sub_feature_map))
		mask = (sub_feature_map != 0)

		np.putmask(_sub_feature_map, mask, bin_feature_map)

		if(np.count_nonzero(_sub_feature_map) != 0):
			sub_ave = np.sum(_sub_feature_map) / np.count_nonzero(_sub_feature_map)
			sub_ave_list.append(sub_ave)
		
		mask_feature_map = scipy.ndimage.zoom(abs(sub_feature_map), 8, order=1)
		
		# Visualize the position in original image.
		# mask = bin_feature_map.reshape(np.shape(original_patch))

		# extent = np.min(original_patch), np.max(original_patch), np.min(mask_feature_map), np.max(mask_feature_map)		# Here is the problem!!!

		# e.set(adjustable='box', aspect='equal')
		input_pathches = original_patch + warped_patch
		im1 = e.imshow(input_pathches, cmap=plt.cm.gray)
		im2 = e.imshow(mask_feature_map, cmap='bwr', alpha=.2)
	
		idx += 1

	plt.subplots_adjust(wspace=-0.15, hspace=-0.15)
	# print(sub_ave_list)
	# if(len(sub_ave_list) == 8):
	# 	print("sus_ave: {}".format(str(np.around(sub_ave_list - H_points))))
	# 	print("prev: {}".format(str(np.around(pred_val - H_points))))

	plt.savefig(save_dir+'/{}hist_map.png'.format(name))
	plt.close('all')
#############################################################
	###################################
	# Get range of the top 30% apparence, visualize a binnary map
	# Only use the channel which have a accurate result (Temply we use 5 pixel, the general loss)
	###################################

	# fig_F, ax_F = plt.subplots(2,4, sharex='all', sharey='all', tight_layout=True, figsize=(12,6))			#
	# idx = 0
	
	# sub_ave_list = []
	# for e in ax_F.reshape(-1):
	# 	if(abs(pred_val[idx]-H_points[idx])>5):				# if the error is too large, do NOT visualize this feature.
	# 		# bin_feature_map = None
	# 		e.text(0.2, 0.2, 'Loss error', ha='center', va='center', fontsize=28, color='C1')
	# 		# idx += 1
	# 		# print("skip!")
	# 		# continue
	# 	bin_feature_map = _feature_map[0,:,:,idx].copy()
	# 	hist_result, bin_edges = np.histogram(bin_feature_map)

	# 	index_hist = np.argsort(hist_result)
	# 	range1 = (bin_edges[index_hist[-1]], bin_edges[index_hist[-1]+1])
	# 	# range2 = (bin_edges[index_hist[-2]], bin_edges[index_hist[-2]+1])
	# 	# range3 = (bin_edges[index_hist[-3]], bin_edges[index_hist[-3]+1])
	# 	# hist_max = np.amax([range1,range2, range3], axis=None)
	# 	# hist_min = np.amin([range1,range2, range3], axis=None)

	# 	hist_max = range1[1]
	# 	hist_min = range1[0]

	# 	# # If the prediction value is NOT in the Tier 1 range, do NOT visualize.
	# 	# if(pred_val[idx]<hist_min or pred_val[idx]>hist_max):
	# 	# 	e.text(0.2, 2, 'Mid error')
	# 	# 	idx += 1
	# 	# 	continue

	# 	# # Only visualize position in the top 1 range, AND if the count is too small, do NOT visualize
	# 	# if(hist_result.max() < 250):
	# 	# 	e.text(0.2, 2, 'Too flat')
	# 	# 	idx += 1
	# 	# 	continue
		
	# 	# print(hist_max, hist_min)

	# 	# bin_feature_map[(hist_max > bin_feature_map) & (bin_feature_map > hist_min)]=255
	# 	# bin_feature_map[bin_feature_map != 255] = 0
	# 	sub_feature_map = bin_feature_map - boardline_feature[0,:,:,idx]
	# 	sub_feature_map = np.around(sub_feature_map, 0)

	# 	_sub_feature_map = np.zeros(np.shape(sub_feature_map))
	# 	mask = (sub_feature_map != 0)

	# 	np.putmask(_sub_feature_map, mask, bin_feature_map)
	# 	# _bin_feature_map[] = bin_feature_map

	# 	if(np.count_nonzero(_sub_feature_map) != 0):
	# 		sub_ave = np.sum(_sub_feature_map) / np.count_nonzero(_sub_feature_map)
	# 		sub_ave_list.append(sub_ave)
		
	# 	mask_feature_map = scipy.ndimage.zoom(abs(sub_feature_map), 8, order=1)
		
	# 	# Visualize the position in original image.
	# 	# mask = bin_feature_map.reshape(np.shape(original_patch))

	# 	# extent = np.min(original_patch), np.max(original_patch), np.min(mask_feature_map), np.max(mask_feature_map)		# Here is the problem!!!

	# 	# e.set(adjustable='box', aspect='equal')
	# 	input_pathches = original_patch + warped_patch
	# 	im1 = e.imshow(input_pathches, cmap=plt.cm.gray)
	# 	im2 = e.imshow(mask_feature_map, cmap='bwr', alpha=.2)
	
	# 	idx += 1

	# plt.subplots_adjust(wspace=-0.15, hspace=-0.15)
	# # print(sub_ave_list)
	# # if(len(sub_ave_list) == 8):
	# # 	print("sus_ave: {}".format(str(np.around(sub_ave_list - H_points))))
	# # 	print("prev: {}".format(str(np.around(pred_val - H_points))))

	# plt.savefig(save_dir+'/{}hist_map.png'.format(name))
	# plt.close('all')
###############################################################

	# Save the matrix
	# f_mat = open(save_dir+'/{}matrix.csv'.format(name), 'w')
	# feature_map_str = _feature_map.astype(str)

	# csv_f = csv.writer(f_mat)
	# for i in range(np.shape(feature_map)[-1]):
	# 	for j in range(np.shape(feature_map)[1]):
	# 		csv_f.writerow(feature_map_str[0,j,:,i])
	# 	csv_f.writerow('')

	# f_mat.close()

	if(len(sub_ave_list) == 8):
		sub_ave_mean = np.mean(abs(sub_ave_list - H_points))

		return sub_ave_mean
	else:
		return 0



def val(val_dir):

	for item in range(1000):
		
		val_image, H_points, base_points, color_image, ori_img, deform_img = get_test(os.path.join(val_dir, 'pure.jpg'))
		save_dir = ''
		for e in H_points:
			save_dir += str(e)
			save_dir += '_'
		save_dir = save_dir[:-1]
		# os.makedirs(save_dir)

		pred, boardline_feature = sess.run([logits, x1], feed_dict={datas: val_image.reshape([1,224,224,2])})
		reference_dir = os.path.join(save_dir, 'reference')
		# os.makedirs(reference_dir)
		# _ = save_fig(pred.astype(int).reshape(-1), reference_dir, ori_img, deform_img, ori_img, deform_img, boardline_feature, H_points, base_points, boardline_feature)

########################
		print(' ========= {} ========= '.format(item))
		err = 0
		err_sub_ave_mean = 0
		cnt_err = 0
		index = 0 

		loc_list = glob(val_dir+'/edge/*')
		cnt_img = len(loc_list) -1

		for i in range(cnt_img):	
			val_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, index, H_points, base_points)
			index += 1
			pred, xx = sess.run([logits, x1], feed_dict={datas: val_images.reshape([1,224,224,2])})
			#print(np.shape(xx))
			pred = np.around(np.array(pred), 2).reshape(-1)
			pred_int = pred.astype(int)
			
			# sub_ave_mean = save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)
			
		################################
		# # Channel normalize for each batch
			# pred_val_idx.append(pred)			#for norm
			# feature_map_idx.append(xx)			#for norm
			
		# #print(np.shape(feature_map_idx))							#(20,1,28,28,8)
		# max_feature = np.amax(feature_map_idx, (0,1,2,3))
		# min_feature = np.amin(feature_map_idx, (0,1,2,3))			#minimum of each channel for 20 patches.
		# feature_range = max_feature - min_feature
		# # print(np.shape(max_feature), min_feature)
		# # quit()

		# pred_idx = 0
		# for feature_map in feature_map_idx:							#(1, 28, 28, 8)
		# 	pred_int = pred_val_idx[pred_idx].astype(int)
		# 	pred_idx += 1
		# 	norm_feature = feature_map.copy()

		# 	for e in range(8):
		# 		norm_feature[0,:,:,e] = (feature_map[0,:,:,e] - min_feature[e])/feature_range[e]
		# 		print(min_feature[e],feature_range[e])

		# 	save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, norm_feature, H_points, base_points)
		#################################

			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			# if(sub_ave_mean != 0):
			# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
			# 	err_sub_ave_mean += sub_ave_mean
			cnt_err += 1

		print('edge_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('edge_focus_mean_loss: {0:.2f}'.format(err_sub_ave_mean/cnt_err))

##########################################
		err = 0
		err_sub_ave_mean = 0
		cnt_err = 0
		index = 0 

		loc_list = glob(val_dir+'/dense/*')
		cnt_img = len(loc_list) -1

		for i in range(cnt_img):	
			val_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, index, H_points, base_points)
			index += 1
			pred, xx = sess.run([logits, x1], feed_dict={datas: val_images.reshape([1,224,224,2])})
			#print(np.shape(xx))
			pred = np.around(np.array(pred), 2).reshape(-1)
			pred_int = pred.astype(int)
			
			# sub_ave_mean = save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)
			
			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			# if(sub_ave_mean != 0):
			# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
			# 	err_sub_ave_mean += sub_ave_mean
			cnt_err += 1

		print('dense_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('dense_focus_mean_loss: {0:.2f}'.format(err_sub_ave_mean/cnt_err))
##########################################
		err = 0
		err_sub_ave_mean = 0
		cnt_err = 0
		index = 0 

		loc_list = glob(val_dir+'/full/*')
		cnt_img = len(loc_list) -1

		for i in range(cnt_img):	
			val_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, index, H_points, base_points)
			index += 1
			pred, xx = sess.run([logits, x1], feed_dict={datas: val_images.reshape([1,224,224,2])})
			#print(np.shape(xx))
			pred = np.around(np.array(pred), 2).reshape(-1)
			pred_int = pred.astype(int)
			
			# sub_ave_mean = save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)
			
			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			# if(sub_ave_mean != 0):
			# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
			# 	err_sub_ave_mean += sub_ave_mean
			cnt_err += 1

		print('full_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('full_focus_mean_loss: {:.2f}'.format(err_sub_ave_mean/cnt_err))




with tf.Graph().as_default():
	datas = tf.placeholder(tf.float32, (None, 224, 224, 2), name='data')
	labels = tf.placeholder(tf.float32, (None, 8), name='label')
	lr = tf.Variable(1e-4, name='learning_rate', trainable=False, dtype=tf.float32)
	
	logits, x1 = homography_regression_model(datas)

	loss = tf.reduce_mean(tf.square(tf.squeeze(logits) - tf.squeeze(labels)))
	
	tf.summary.scalar('loss',loss)

	opt = tf.train.AdamOptimizer(lr).minimize(loss)

	saver = tf.train.Saver(max_to_keep=None)
		
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		saver.restore(sess, './H_predict/ckpt/baseline/model499.ckpt')
		
		merged = tf.summary.merge_all()  
		writer = tf.summary.FileWriter(log_dir,sess.graph)  
		saver_all = tf.train.Saver(max_to_keep=1000)

		'''
		
		queue = Queue(maxsize = 32)
		processes = [Process(target = get_generator, args = (queue, "./train/*.jpg", batch_size)) for x in range(4)]

		for p in processes:
			p.start()
		for e in range(epochs):
			for item in range(100):
				batch_x, batch_y = consumer(queue) #next(g)
				summary, _, cl = sess.run([merged, opt, loss], feed_dict={datas: batch_x, labels: batch_y})
				
			logging.info(' ========= {} ========= '.format(e))
			logging.info('training_loss: {}'.format(cl))
			
			if((e+1)%100 == 0):
				save_path = saver_all.save(sess, "{}/model{}.ckpt".format(ckpt_dir, e))
				print ('Model saved in file: %s' % save_path)
				
				err = 0
				for item in range(100):
					val_image, H_points, base_points, color_image, ori_img, deform_img = get_test("./test/*.jpg")
					pred = sess.run([logits], feed_dict={datas: val_image.reshape([1,224,224,2])})
					pred = np.array(pred)#.reshape((4,2))
					err += np.mean(np.abs(pred-H_points))
				logging.info('testing_loss: {}'.format(err/100))
	
		queue.close()
		for p in processes:
			p.terminate()
		quit()
		'''

###testing###
		fig=plt.figure(figsize=(11,8)) 

		val("./val_edge")
		'''
		# rand_list = ((8327, 40, 59, 31, -10),(57231, 52, 56, 2, -12),(39447, 59, 38, -31, 17),(25028, 36, 37, -23, 12),(65169, 40, 34, 2, 20),(36079, 59, 40, -2, -7),(74480, 54, 52, 19, -10),(63721, 64, 63, 29, 32),(64076, 52, 35, -13, -22),(80661, 36, 52, 3, -26),(48121, 41, 40, 7, -31),(72510, 61, 55, 24, 29),(82010, 62, 54, -6, -28),(25491, 56, 60, 13, -7),(13841, 50, 52, 11, 3),(27017, 53, 47, 16, -28),(54712, 60, 48, -21, 26),(713, 53, 35, 19, 0),(53564, 51, 40, 26, 29),(35527, 38, 55, 20, 7))
		for item in range(100):
			
			val_image, H_points, base_points, color_image, ori_img, deform_img = get_test("./test/*")
			# get_test_demo("/home/shao/Documents/DATASETS/BSDS300/images/train/*.jpg", rand_list[item])
			# get_test("/home/shao/Documents/DATASETS/VOC2011/train/*.jpg")
			# get_test("/home/shao/Documents/DATASETS/COCO/images/*.jpg")

			
			pred = sess.run([logits], feed_dict={datas: val_image.reshape([1,224,224,2])})
			pred = np.array(pred)#.reshape((4,2))

			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			print(err/100)
			
			# ###err image###
			
			# # ###		THE INITIAL POINTS AFFECT THE RESULT	###
			# x = y = 0
			# top_left_point = (x, y)
			# bottom_left_point = (224 + x, y)
			# bottom_right_point = (224 + x, 224 + y)
			# top_right_point = (x, 224 + y)
			# four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
			# four_points = np.array(four_points)
			
			# H_ = cv2.getPerspectiveTransform(np.float32(four_points).reshape((4,2)), np.float32(four_points+pred.reshape((4,2))))
			# H_inverse_ = inv(H_)
			# img3_sftd = cv2.warpPerspective(val_image[:,:,0], H_inverse_, (224, 224))
			
			# ##################	In order to transform the whole image, the base_points on the original image is necessary, if just transfer the cropped patch
			# ### just define as (0,0)-(224,224)
			label_4point = (H_points + base_points).reshape((4,2))	
			pred_4point = np.reshape((pred + base_points),(4,2))

			H = cv2.getPerspectiveTransform(np.float32(base_points).reshape((4,2)), np.float32(label_4point))
			H_inverse = inv(H)

			# # ################

			# img2_sftd = cv2.warpPerspective(cv2.resize(cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY),(320,320)), H_inverse, (320, 320))
			# img2_sftd = img2_sftd[base_points[1]:base_points[5], base_points[0]:base_points[4]]

			# print(pred)

			# ax=plt.subplot(2,2,3)
			# plt.title("Input_1")
			# ax.imshow(cv2.resize(val_image[:,:,0], (224, 224)),cmap='gray')

			# ax=plt.subplot(2,2,4)
			# plt.title("Input_2")
			# ax.imshow(cv2.resize(val_image[:,:,1], (224, 224)),cmap='gray')


			# # ax=plt.subplot(3,2,3)
			# # ax.imshow(img2_sftd, cmap='gray')

			# ax=plt.subplot(2,2,3)
			# plt.title("Transform")
			# ax.imshow(img3_sftd, cmap='gray')
			
			# # ax=plt.subplot(3,2,5)
			# # ax.imshow(img2_sftd)
			# # err = img2_sftd-val_image[:,:,1]		#use it to compare the 2 transform works in same way
			# # im=ax.imshow(err, vmin=-25.5, vmax=25.5, cmap='jet')
			# # plt.colorbar(im,fraction=0.046, pad=0.04)


			# ax=plt.subplot(2,2,4)
			# plt.title("Diff")
			# err = img3_sftd-val_image[:,:,1]		#the resize is a problem cause the err doesn't work
			# # # im=ax.imshow(err, vmin=-.25, vmax=.25)
			# im=ax.imshow(err, vmin=-25.5, vmax=25.5, cmap='jet')
			# plt.colorbar(im,fraction=0.046, pad=0.04)
			# plt.show()
			# quit()

			
			###showing case###

			transform_img = cv2.warpPerspective(cv2.resize(ori_img,(320,320)), H_inverse, (320, 320))
			# transform_img = deform_img 
			transform_img_ = transform_img.copy()
			color_image = ori_img 

			ax=plt.subplot(2,3,1)
			# plt.title("ori_img")
			color_image_draw = cv2.polylines(color_image, np.array([base_points.reshape((4,2))], np.int32), 1, (255,0,0), 2)
			ax.imshow(color_image_draw, vmin=0, vmax=255, cmap='jet')

			
			label_4point = (-H_points + base_points).reshape((4,2))	
			pred_4point = np.int32(np.reshape((-pred + base_points),(4,2)))
			# print(pred_4point, label_4point)

			cv2.polylines(transform_img, np.array([label_4point], np.int32), 1, (255,0,0), 2)
			# cv2.polylines(transform_img, np.array([pred_4point], np.int32), 1, (0,0,255), 2)
			# cv2.polylines(transform_img, np.array([base_points.reshape((4,2))], np.int32), 1, (0,0,255), 2)
			# mce = np.sum(abs(pred-H_points))/8
			# print(mce)

			# cv2.putText(transform_img,'Mean Corner Error = {0:.2f}'.format(mce),(20,20), 4, 0.5, (255,0,0),1,cv2.LINE_AA)

			ax=plt.subplot(2,3,2)
			# plt.title("transform_img")
			ax.imshow(transform_img, vmin=0, vmax=255, cmap='jet')

			ax=plt.subplot(2,3,3)

			cv2.polylines(transform_img_, np.array([base_points.reshape((4,2))], np.int32), 1, (0,0,255), 2)
			# plt.title("transform_img")
			ax.imshow(transform_img_, vmin=0, vmax=255, cmap='jet')

			# plt.subplots_adjust(wspace=-0.35)
			plt.savefig('{}.png'.format(item))
			
			plt.show()
			
		#quit()

		'''




