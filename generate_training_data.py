import cv2

import numpy as np
#import matplotlib.pyplot as plt

import deformation


#########################################
# CV2 based python 
#
# Mingzhen Shao
# Oct 28, 2021
#########################################


size_img = 256

def load_img(img_path):
	
	try:
		img = cv2.imread(img_path)
	except Exception as e:
		print(e)
		return

	#print(img.shape)
	img_resize = cv2.resize(img, (size_img, size_img))
	return img_resize


def light_adj(img_input):			# change the light/dark parts, or normalization 

	light_adj_threshold_high = 230
	light_adj_threshold_low = 30

	##########################
	# normalization

	#norm = np.zeros((size_img, size_img))
	#img_normalize = cv2.normalize(img_input, None, 0, 255, cv2.NORM_MINMAX) 
	#img = img_normalize

	###########################
	#lighe_adj


	img_hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV).astype("float32")
	(h, s, v) = cv2.split(img_hsv)

	v[v > light_adj_threshold_high] += 10
	v[v <= light_adj_threshold_low] -= 10

	v[v>255] = 255
	v[v<0] = 0

	#s = s * scale
	#s = np.clip(s, 0, 255)
	img_hsv = cv2.merge([h, s, v])

	img = cv2.cvtColor(img_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

	return img


def color_adj(img_input):			# slight adj the H channel 
	
	H_adj = 30 
	S_adj = 0.1

	img_hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV).astype("float32")
	(h, s, v) = cv2.split(img_hsv)

	h += 30
	h[h>360] -= 360

	#s = s * scale
	#s = np.clip(s, 0, 255)
	img_hsv = cv2.merge([h, s, v])

	img = cv2.cvtColor(img_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

	return img


def cover_adj(img_input):		# the size of the covered part should be less than 1/16 of the whole image, random location, (random shape)

	covered_img_num = np.random.randint(0,365) 
	covered_img_path = '/home/mshao/Downloads/img/' + str(covered_img_num) +'.jpg'
	covered_img = cv2.imread(covered_img_path)

	print(covered_img_path)

	covered_size = int(size_img / (4*np.random.randint(1,9)))
	covered_locate = (np.random.randint(covered_size, size_img-covered_size), np.random.randint(covered_size, size_img-covered_size))
	#covered_rotate 

	covered_img_resize = cv2.resize(covered_img, (covered_size, covered_size))

	img[covered_locate[0]:covered_locate[0]+covered_size, covered_locate[1]:covered_locate[1]+covered_size] = covered_img_resize

	return img 


def distortion_adj(img):		#
	pass
	#tmp_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	pre, out = deformation.Deform(img, deform_size=20, img_size=512)

	return pre, out


def generating_simple_shapes():

	# image = np.zeros((320,320), np.uint8)

	# Center coordinates
	center_coordinates = (160, 160)
	 
	# Radius of circle
	radius = 60
	# color = (255)
	  
	# Line thickness of -1 px
	thickness = 1
	  
	# Using cv2.circle() method
	# Draw a circle of red color of thickness -1 px
	
	for e in range(10):
		image = np.zeros((320,320), np.uint8)
		color = 255

		x1 = np.random.randint(40,160)
		y1 = np.random.randint(40,160)
		x2 = np.random.randint(160,280)
		y2 = np.random.randint(160,280)

		for i in range(0, 150, 50):
			color -= i 

			image = cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
			# image = cv2.circle(image, center_coordinates, radius, color, thickness)

			# triangle_cnt = np.array( [(160,60), (80,130), (110,220), (170,220), (220, 125)] )

			# image = cv2.drawContours(image, [triangle_cnt], 0, color, -1)
			
		
			cv2.imwrite('Rect_{}_{}.jpg'.format(e,color), image)
		
def generating_line():
	image = np.zeros((320,320), np.uint8)
	color = 255
	for i in range(0, 150, 20):
		color -= i 
		
		# image = cv2.line(image, (70, 120), (140, 200), color, 4)
		# image = np.full((320,320), color)
		image = cv2.circle(image, (90,110), 1, color, -1)
		image = cv2.circle(image, (80,250), 1, color, -1)
		image = cv2.circle(image, (140,70), 1, color, -1)
		image = cv2.circle(image, (210,260), 1, color, -1)
	
		cv2.imwrite('point4_3_{}.jpg'.format(color), image)


def generating_pair_rect():

	for e in range(30):
		
		x1 = np.random.randint(30,120)
		y1 = np.random.randint(30,120)
		x2 = np.random.randint(200,290)
		y2 = np.random.randint(200,290)

		for color in range(255, 55, -50):
			# frame
			image_rect = np.zeros((320,320), np.uint8)
			image_rect = cv2.rectangle(image_rect, (x1,y1), (x2,y2), color, 1)
			cv2.imwrite('rect_{}_{}.jpg'.format(e, color), image_rect)


		# dense frame
			image_dense = np.zeros((320,320), np.uint8)
			diff_x = x2 - x1
			diff_y = y2 - y1
			for i in range(3):
				biase_x = int(i/3.0 * diff_x/2)
				biase_y = int(i/3.0 * diff_y/2)	
				# print(biase_x, biase_y)
				image_rect = cv2.rectangle(image_dense, ((x1+biase_x),(y1+biase_y)), ((x2 - biase_x), (y2 - biase_y)), color, 1)


			cv2.imwrite('rect_dense_{}_{}.jpg'.format(e, color), image_dense)

		# fulled frame
			image_fulled = np.zeros((320,320), np.uint8)
			image_fulled = cv2.rectangle(image_fulled, (x1,y1), (x2,y2), color, -1)
			cv2.imwrite('rect_fulled_{}_{}.jpg'.format(e, color), image_fulled)


def generating_pair_circle():

	for e in range(30):
		
		x = np.random.randint(80,260)
		y = np.random.randint(80,260)
		radius = np.random.randint(30,70)
		center_coordinates = (x, y)

		for color in range(255, 55, -50):
			# frame
			image_rect = np.zeros((320,320), np.uint8)
			image_rect = cv2.circle(image_rect, center_coordinates, radius, color, 1)
			cv2.imwrite('circle_{}_{}.jpg'.format(e, color), image_rect)


		# dense frame
			image_dense = np.zeros((320,320), np.uint8)
		
			for i in range(3):
				biase_r = int(i/3.0 * radius)
				
				image_rect = cv2.circle(image_dense, center_coordinates, (radius - biase_r), color, 1)

			cv2.imwrite('circle_dense_{}_{}.jpg'.format(e, color), image_dense)

		# fulled frame
			image_fulled = np.zeros((320,320), np.uint8)
			image_fulled = cv2.circle(image_fulled, center_coordinates, radius, color, -1)
			cv2.imwrite('circle_fulled_{}_{}.jpg'.format(e, color), image_fulled)




#Final step of image pre-process, turn into 0-1 float array
def img_norm(img):
	img_array = np.asarray(img)
	img_array_norm = img_array / 255.0

	return img_array_norm


if __name__ == '__main__':
	# img = load_img('./20.jpg')
	# #img_1 = cover_adj(img)
	# pre, out = distortion_adj(img)
	# cv2.imwrite('pre.jpg', pre)
	# cv2.imwrite('out.jpg', out)

	# generating_simple_shapes()
	generating_pair_rect()
	generating_pair_circle()


	cv2.waitKey(50)