import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cv2, random, sys
#import oct
#from skimage.transform import resize

np.set_printoptions(threshold=sys.maxsize)


N=12

basicMx = np.random.randint(-deform_size, deform_size, (N,N))
basicMy = np.random.randint(-deform_size, deform_size, (N,N))

# basicMx[0][:] = 0
# basicMx[-1][:] = 0
# basicMx.T[0][:] = 0
# basicMx.T[-1][:] = 0
# basicMy[0][:] = 0
# basicMy[-1][:] = 0
# basicMy.T[0][:] = 0
# basicMy.T[-1][:] = 0


# tmp_img = oct.ConvertH2CBGR2Gray(cv2.imread("etoct.jpg"))

tmp_img = cv2.imread("cover_adj.jpg")
tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)
tmp_img = cv2.resize(tmp_img,(512,512))


tmp_img_ = tmp_img.copy()


basicMxAux = basicMx.reshape(N**2)

# ijk is an (N**3,3) array with the indexes of the reshaped array.
ijk = np.mgrid[0:N,0:N].reshape(2,N**2).T
n = 512j
i,j = np.mgrid[0:N:n,0:N:n]

tmpx = griddata(ijk,basicMxAux,(i,j),method="cubic")
tmpx[np.isnan(tmpx)] = 0

# plt.subplot(224)
# plt.imshow(tmpx)
# plt.show()


basicMyAux = basicMy.reshape(N**2)
tmpy = griddata(ijk,basicMyAux,(i,j),method="cubic")
tmpy[np.isnan(tmpy)] = 0

tmpx = np.int8(tmpx)
tmpy = np.int8(tmpy)


# fig1, ax1 = plt.subplots()
# ax1.set_title('Arrows scale with plot width, not view')
# Q = ax1.quiver(i, j, tmpx, tmpy, units='width')
# qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')

fig2, ax2 = plt.subplots()
ax2.set_title("pivot='mid'; every third arrow; units='inches'")
Q = ax2.quiver(i[::20, ::20], j[::20, ::20], tmpx[::20, ::20], tmpy[::20, ::20],
               pivot='mid', units='inches')
qk = ax2.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
# ax2.scatter(i[::10, ::10], j[::10, ::10], color='r', s=5)
# plt.show()


for i  in range(512):
	for j in range(512):

		tmp_img_[i][j] = tmp_img[i+tmpx[i][j]][j + tmpy[i][j]]
##### CROP #####

out = tmp_img_[64:448, 64:448]
pre = tmp_img[64:448, 64:448]

fig=plt.figure(figsize=(11,8)) 
plt.subplot(231)
plt.imshow(tmp_img,vmin=0, vmax=255, cmap='jet')
plt.subplot(232)
plt.imshow(tmp_img_,vmin=0, vmax=255,cmap='jet')
plt.subplot(233)
fig = plt.imshow(np.int8(tmp_img) - np.int8(tmp_img_), vmin=-25, vmax=25, cmap='jet')
plt.colorbar(fig,fraction=0.046, pad=0.04)

plt.subplot(234)
plt.imshow(pre, vmin=0, vmax=255,cmap='jet')
plt.subplot(235)
plt.imshow(out, vmin=0, vmax=255,cmap='jet')
plt.subplot(236)
fig = plt.imshow(np.int8(pre) - np.int8(out),vmin=-25, vmax=25, cmap='jet')
plt.colorbar(fig,fraction=0.046, pad=0.04)
# plt.subplot(224)
# plt.imshow(tmp)
plt.savefig('etcot_demo.png')
plt.show()



def deformation(tmp_img, deform_size=20, img_size=512):
	N=12

	basicMx = np.random.randint(-deform_size, deform_size, (N,N))
	basicMy = np.random.randint(-deform_size, deform_size, (N,N))

	# tmp_img = oct.ConvertH2CBGR2Gray(cv2.imread("tmp.png"))

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


