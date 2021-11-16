import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cv2, random, sys
#from skimage.transform import resize
#import oct

np.set_printoptions(threshold=sys.maxsize)


def Deform(tmp_img, deform_size=20, img_size=512):

	N=12
	basicMx = []
	basicMy = []
	for i  in range(12):
		for j in range(12):
			basicMx.append((random.randint(-deform_size, deform_size)))
			basicMy.append((random.randint(-deform_size, deform_size)))

	basicMx = np.reshape(basicMx, (12,12))
	basicMy = np.reshape(basicMy, (12,12))

	# tmp_img = oct.ConvertH2CBGR2Gray(cv2.imread("tmp.png"))
	tmp_img = cv2.resize(tmp_img, (512, 512))

	# tmp_img = cv2.imread("tmp.jpg", cv2.COLOR_RGB2GRAY)
	tmp_img_ = tmp_img.copy()

	basicMxAux = basicMx.reshape(N**2)

	# ijk is an (N**3,3) array with the indexes of the reshaped array.
	ijk = np.mgrid[0:N,0:N].reshape(2,N**2).T
	n = 512j
	i,j = np.mgrid[0:N:n,0:N:n]

	tmpx = griddata(ijk,basicMxAux,(i,j),method="cubic")
	tmpx[np.isnan(tmpx)] = 0

	basicMyAux = basicMy.reshape(N**2)
	tmpy = griddata(ijk,basicMyAux,(i,j),method="cubic")
	tmpy[np.isnan(tmpy)] = 0

	tmpx = np.int8(tmpx)
	tmpy = np.int8(tmpy)

	for i  in range(img_size):
		for j in range(img_size):
			tmp_img_[i][j] = tmp_img[i+tmpx[i][j]][j + tmpy[i][j]]
	##### CROP #####
	pre = tmp_img[64:448, 64:448]
	out = tmp_img_[64:448, 64:448]
	
	return pre, out


