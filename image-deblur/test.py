import cv2 
import pylops
import numpy as np
# import an image to np array 

# im_frame = Image.open("pika.png")
# np_frame = np.array(im_frame.getdata())

im = cv2.imread("buzz.png")[:, :, 0]
cv2.imwrite("buzz_src.png", im)
Nz, Nx = im.shape 
print(Nz)
print(Nx)

# blurring Gaussian operator 
nh = [15, 25]
hz = np.exp(-0.1*np.linspace(-(nh[0]//2), nh[0]//2, nh[0])**2)
hx = np.exp(-0.03*np.linspace(-(nh[1]//2), nh[1]//2, nh[1])**2)
hz /= np.trapz(hz) # normalize the integral to 1
hx /= np.trapz(hx) # normalize the integral to 1
h = hz[:, np.newaxis] * hx[np.newaxis, :]
print(hz[:, np.newaxis])
print(hx[np.newaxis, :])
print(h.shape)

Cop = pylops.signalprocessing.Convolve2D(Nz * Nx, h=h,
                                         offset=(nh[0] // 2,
                                                 nh[1] // 2),
                                         dims=(Nz, Nx), dtype='float32')
        
# imblur = Cop * im.flatten()
# 
# imdeblur = pylops.optimization.sparsity.FISTA(Cop, imblur, 50, alpha=.5,
#                                           show=True, returninfo=True)
# 
# # imdeblur = pylops.optimization.leastsquares.NormalEquationsInversion(Cop, None,
# #                                                               imblur,
# #                                                               maxiter=50, returninfo=True)
# 
# print(imdeblur[1])
# imblur = imblur.reshape((Nz, Nx))
# imdeblur = imdeblur[0].reshape((Nz, Nx))
# 
# cv2.imwrite("buzz_blurred.png", imblur)
# cv2.imwrite("buzz_deblurred_FISTA.png", imdeblur)
