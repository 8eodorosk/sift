import numpy as np
import scipy
from scipy import signal
from scipy import misc
from scipy import ndimage
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy.linalg
import matplotlib.pyplot as plt
import sys



def detect_keypoints(imagename, threshold):
	original = ndimage.imread(imagename, flatten=True)
	# SIFT Parameters
	#threshold = 4
	s = 3
	k = 2 ** (1.0 / s)
	# threshold variable is the contrast threshold. Set to at least 1

	# Standard deviations for Gaussian smoothing
	kvec1 = np.array([1, 1.3, 1.6 , 1.6 * k , 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5)])
	kvec2 = np.array([1.6 * k , 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6),1.6 * (k ** 7), 1.6 * (k ** 8)])
	kvec3 = np.array([1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])
	kvec4 = np.array([1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11), 1.6 * (k ** 12), 1.6 * (k ** 13), 1.6 * (k ** 14)])
	kvec5 = np.array([1.6 * (k ** 10), 1.6 * (k ** 11), 1.6 * (k ** 12), 1.6 * (k ** 13), 1.6 * (k ** 14), 1.6 * (k ** 15), 1.6 * (k ** 16), 1.6 * (k ** 17)])




	kvectotal = np.array([1.3, 1.6 , 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3),
	                      1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6),1.6 * (k ** 7),1.6 * (k ** 8), 
	                      1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11),1.6 * (k ** 12), 1.6 * (k ** 13),
	                      1.6 * (k ** 14), 1.6 * (k ** 15), 1.6 * (k ** 16), 1.6 * (k ** 17), 1.6 * (k ** 18),
	                      1.6 * (k ** 19), 1.6 * (k ** 20), 1.6 * (k ** 21), 1.6 * (k ** 22), 1.6 * (k ** 23)])





	# Downsampling images
	doubled = misc.imresize(original, 200, 'bilinear').astype(int)
	normal = misc.imresize(doubled, 50, 'bilinear').astype(int)
	halved = misc.imresize(normal, 50, 'bilinear').astype(int)
	quartered = misc.imresize(halved, 50, 'bilinear').astype(int)
	eighted = misc.imresize(quartered, 50, 'bilinear').astype(int)


	# Initialize Gaussian pyramids
	pyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 8))
	pyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 8))
	pyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 8))
	pyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 8))
	pyrlvl5 = np.zeros((eighted.shape[0], eighted.shape[1], 8))


	# Construct Gaussian pyramids
	for i in range(0, 8):
	    pyrlvl1[:,:,i] = ndimage.filters.gaussian_filter(doubled, kvec1[i])   
	    pyrlvl2[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec2[i]), 50, 'bilinear') 
	    pyrlvl3[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec3[i]), 25, 'bilinear')
	    pyrlvl4[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec4[i]), 1.0 / 8.0, 'bilinear')
	    pyrlvl5[:,:,i] = misc.imresize(ndimage.filters.gaussian_filter(doubled, kvec5[i]), 1.0 / 16.0, 'bilinear')

	 # Initialize Difference-of-Gaussians (DoG) pyramids
	diffpyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 7))
	diffpyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 7))
	diffpyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 7))
	diffpyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 7))
	diffpyrlvl5 = np.zeros((eighted.shape[0], eighted.shape[1], 7))


	# Construct DoG pyramids
	for i in range(0, 7):
	    diffpyrlvl1[:,:,i] = pyrlvl1[:,:,i+1] - pyrlvl1[:,:,i]
	    diffpyrlvl2[:,:,i] = pyrlvl2[:,:,i+1] - pyrlvl2[:,:,i]
	    diffpyrlvl3[:,:,i] = pyrlvl3[:,:,i+1] - pyrlvl3[:,:,i]
	    diffpyrlvl4[:,:,i] = pyrlvl4[:,:,i+1] - pyrlvl4[:,:,i]
	    diffpyrlvl5[:,:,i] = pyrlvl5[:,:,i+1] - pyrlvl5[:,:,i]

	# Initialize pyramids to store extrema locations
	extrpyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
	extrpyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 5))
	extrpyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 5))
	extrpyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 5))
	extrpyrlvl5 = np.zeros((eighted.shape[0], eighted.shape[1], 5))

	print "Starting extrema detection..."
	print "First octave"

	for i in range(1, 6):
	    for j in range(80, doubled.shape[0] - 80):
	        for k in range(80, doubled.shape[1] - 80):
	            if np.absolute(diffpyrlvl1[j, k, i]) < threshold:
	                continue  

	            maxbool = (diffpyrlvl1[j, k, i] > 0)
	            #print ("max bool for iteration: " ,i,j,k , "is" , maxbool , "\n")
	            minbool = (diffpyrlvl1[j, k, i] < 0)
	            #print ("minbool for iteration: " ,i,j,k , "is" , minbool , "\n")

	            for di in range(-1, 2):
	                for dj in range(-1, 2):
	                    for dk in range(-1, 2):
	                        #print (di,dj,dk,"\n")
	                        if di == 0 and dj == 0 and dk == 0:
	                            continue
	                        #print(i,j,k, "times" ,diffpyrlvl1[j, k, i], diffpyrlvl1[j + dj, k + dk, i + di],j + dj, k + dk, i + di ,"\n")
	                        maxbool = maxbool and (diffpyrlvl1[j, k, i] > diffpyrlvl1[j + dj, k + dk, i + di])
	                        minbool = minbool and (diffpyrlvl1[j, k, i] < diffpyrlvl1[j + dj, k + dk, i + di])
	                        if not maxbool and not minbool:
	                            break

	                    if not maxbool and not minbool:
	                        break

	                if not maxbool and not minbool:
	                    break

	            if maxbool or minbool:
	                dx = (diffpyrlvl1[j, k+1, i] - diffpyrlvl1[j, k-1, i]) * 0.5 / 255
	                dy = (diffpyrlvl1[j+1, k, i] - diffpyrlvl1[j-1, k, i]) * 0.5 / 255
	                ds = (diffpyrlvl1[j, k, i+1] - diffpyrlvl1[j, k, i-1]) * 0.5 / 255
	                dxx = (diffpyrlvl1[j, k+1, i] + diffpyrlvl1[j, k-1, i] - 2 * diffpyrlvl1[j, k, i]) * 1.0 / 255        
	                dyy = (diffpyrlvl1[j+1, k, i] + diffpyrlvl1[j-1, k, i] - 2 * diffpyrlvl1[j, k, i]) * 1.0 / 255          
	                dss = (diffpyrlvl1[j, k, i+1] + diffpyrlvl1[j, k, i-1] - 2 * diffpyrlvl1[j, k, i]) * 1.0 / 255
	                dxy = (diffpyrlvl1[j+1, k+1, i] - diffpyrlvl1[j+1, k-1, i] - diffpyrlvl1[j-1, k+1, i] + diffpyrlvl1[j-1, k-1, i]) * 0.25 / 255 
	                dxs = (diffpyrlvl1[j, k+1, i+1] - diffpyrlvl1[j, k-1, i+1] - diffpyrlvl1[j, k+1, i-1] + diffpyrlvl1[j, k-1, i-1]) * 0.25 / 255 
	                dys = (diffpyrlvl1[j+1, k, i+1] - diffpyrlvl1[j-1, k, i+1] - diffpyrlvl1[j+1, k, i-1] + diffpyrlvl1[j-1, k, i-1]) * 0.25 / 255  

	                dD = np.matrix([[dx], [dy], [ds]])
	                H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
	                x_hat = numpy.linalg.lstsq(H, dD)[0]
	                D_x_hat = diffpyrlvl1[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

	                r = 10.0
	                if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):
	                    extrpyrlvl1[j, k, i - 1] = 1
	                    
	print "Second octave"

	for i in range(1, 6):
	    for j in range(40, normal.shape[0] - 40):
	        for k in range(40, normal.shape[1] - 40):
	            if np.absolute(diffpyrlvl2[j, k, i]) < threshold:
	                continue   

	            maxbool = (diffpyrlvl2[j, k, i] > 0)
	            minbool = (diffpyrlvl2[j, k, i] < 0)

	            for di in range(-1, 2):
	                for dj in range(-1, 2):
	                    for dk in range(-1, 2):
	                        if di == 0 and dj == 0 and dk == 0:
	                            continue
	                        maxbool = maxbool and (diffpyrlvl2[j, k, i] > diffpyrlvl2[j + dj, k + dk, i + di])
	                        minbool = minbool and (diffpyrlvl2[j, k, i] < diffpyrlvl2[j + dj, k + dk, i + di])
	                        if not maxbool and not minbool:
	                            break

	                    if not maxbool and not minbool:
	                        break

	                if not maxbool and not minbool:
	                    break

	            if maxbool or minbool:
	                dx = (diffpyrlvl2[j, k+1, i] - diffpyrlvl2[j, k-1, i]) * 0.5 / 255
	                dy = (diffpyrlvl2[j+1, k, i] - diffpyrlvl2[j-1, k, i]) * 0.5 / 255
	                ds = (diffpyrlvl2[j, k, i+1] - diffpyrlvl2[j, k, i-1]) * 0.5 / 255
	                dxx = (diffpyrlvl2[j, k+1, i] + diffpyrlvl2[j, k-1, i] - 2 * diffpyrlvl2[j, k, i]) * 1.0 / 255        
	                dyy = (diffpyrlvl2[j+1, k, i] + diffpyrlvl2[j-1, k, i] - 2 * diffpyrlvl2[j, k, i]) * 1.0 / 255          
	                dss = (diffpyrlvl2[j, k, i+1] + diffpyrlvl2[j, k, i-1] - 2 * diffpyrlvl2[j, k, i]) * 1.0 / 255
	                dxy = (diffpyrlvl2[j+1, k+1, i] - diffpyrlvl2[j+1, k-1, i] - diffpyrlvl2[j-1, k+1, i] + diffpyrlvl2[j-1, k-1, i]) * 0.25 / 255 
	                dxs = (diffpyrlvl2[j, k+1, i+1] - diffpyrlvl2[j, k-1, i+1] - diffpyrlvl2[j, k+1, i-1] + diffpyrlvl2[j, k-1, i-1]) * 0.25 / 255 
	                dys = (diffpyrlvl2[j+1, k, i+1] - diffpyrlvl2[j-1, k, i+1] - diffpyrlvl2[j+1, k, i-1] + diffpyrlvl2[j-1, k, i-1]) * 0.25 / 255  

	                dD = np.matrix([[dx], [dy], [ds]])
	                H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
	                x_hat = numpy.linalg.lstsq(H, dD)[0]
	                D_x_hat = diffpyrlvl2[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

	                r = 10.0
	                if (((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(x_hat[0]) < 0.5 and np.absolute(x_hat[1]) < 0.5 and np.absolute(x_hat[2]) < 0.5 and np.absolute(D_x_hat) > 0.03:
	                    extrpyrlvl2[j, k, i - 1] = 1
	                    
	print "Third octave"

	for i in range(1, 6):
	    for j in range(20, halved.shape[0] - 20):
	        for k in range(20, halved.shape[1] - 20):
	            if np.absolute(diffpyrlvl3[j, k, i]) < threshold:
	                continue  

	            maxbool = (diffpyrlvl3[j, k, i] > 0)
	            minbool = (diffpyrlvl3[j, k, i] < 0)

	            for di in range(-1, 2):
	                for dj in range(-1, 2):
	                    for dk in range(-1, 2):
	                        if di == 0 and dj == 0 and dk == 0:
	                            continue
	                        maxbool = maxbool and (diffpyrlvl3[j, k, i] > diffpyrlvl3[j + dj, k + dk, i + di])
	                        minbool = minbool and (diffpyrlvl3[j, k, i] < diffpyrlvl3[j + dj, k + dk, i + di])
	                        if not maxbool and not minbool:
	                            break

	                    if not maxbool and not minbool:
	                        break

	                if not maxbool and not minbool:
	                    break

	            if maxbool or minbool:
	                dx = (diffpyrlvl3[j, k+1, i] - diffpyrlvl3[j, k-1, i]) * 0.5 / 255
	                dy = (diffpyrlvl3[j+1, k, i] - diffpyrlvl3[j-1, k, i]) * 0.5 / 255
	                ds = (diffpyrlvl3[j, k, i+1] - diffpyrlvl3[j, k, i-1]) * 0.5 / 255
	                dxx = (diffpyrlvl3[j, k+1, i] + diffpyrlvl3[j, k-1, i] - 2 * diffpyrlvl3[j, k, i]) * 1.0 / 255        
	                dyy = (diffpyrlvl3[j+1, k, i] + diffpyrlvl3[j-1, k, i] - 2 * diffpyrlvl3[j, k, i]) * 1.0 / 255          
	                dss = (diffpyrlvl3[j, k, i+1] + diffpyrlvl3[j, k, i-1] - 2 * diffpyrlvl3[j, k, i]) * 1.0 / 255
	                dxy = (diffpyrlvl3[j+1, k+1, i] - diffpyrlvl3[j+1, k-1, i] - diffpyrlvl3[j-1, k+1, i] + diffpyrlvl3[j-1, k-1, i]) * 0.25 / 255 
	                dxs = (diffpyrlvl3[j, k+1, i+1] - diffpyrlvl3[j, k-1, i+1] - diffpyrlvl3[j, k+1, i-1] + diffpyrlvl3[j, k-1, i-1]) * 0.25 / 255 
	                dys = (diffpyrlvl3[j+1, k, i+1] - diffpyrlvl3[j-1, k, i+1] - diffpyrlvl3[j+1, k, i-1] + diffpyrlvl3[j-1, k, i-1]) * 0.25 / 255  

	                dD = np.matrix([[dx], [dy], [ds]])
	                H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
	                x_hat = numpy.linalg.lstsq(H, dD)[0]
	                D_x_hat = diffpyrlvl3[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

	                r = 10.0
	                if (((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(x_hat[0]) < 0.5 and np.absolute(x_hat[1]) < 0.5 and np.absolute(x_hat[2]) < 0.5 and np.absolute(D_x_hat) > 0.03:
	                    #print("MPIKA")
	                    extrpyrlvl3[j, k, i - 1] = 1
	                    
	                    
	print "Fourth octave"

	for i in range(1, 6):
	    for j in range(10, quartered.shape[0] - 10):
	        for k in range(10, quartered.shape[1] - 10):
	            if np.absolute(diffpyrlvl4[j, k, i]) < threshold:
	                continue

	            maxbool = (diffpyrlvl4[j, k, i] > 0)
	            minbool = (diffpyrlvl4[j, k, i] < 0)

	            for di in range(-1, 2):
	                for dj in range(-1, 2):
	                    for dk in range(-1, 2):
	                        if di == 0 and dj == 0 and dk == 0:
	                            continue
	                        maxbool = maxbool and (diffpyrlvl4[j, k, i] > diffpyrlvl4[j + dj, k + dk, i + di])
	                        minbool = minbool and (diffpyrlvl4[j, k, i] < diffpyrlvl4[j + dj, k + dk, i + di])
	                        if not maxbool and not minbool:
	                            break

	                    if not maxbool and not minbool:
	                        break

	                if not maxbool and not minbool:
	                    break

	            if maxbool or minbool:
	                dx = (diffpyrlvl4[j, k+1, i] - diffpyrlvl4[j, k-1, i]) * 0.5 / 255
	                dy = (diffpyrlvl4[j+1, k, i] - diffpyrlvl4[j-1, k, i]) * 0.5 / 255
	                ds = (diffpyrlvl4[j, k, i+1] - diffpyrlvl4[j, k, i-1]) * 0.5 / 255
	                dxx = (diffpyrlvl4[j, k+1, i] + diffpyrlvl4[j, k-1, i] - 2 * diffpyrlvl4[j, k, i]) * 1.0 / 255        
	                dyy = (diffpyrlvl4[j+1, k, i] + diffpyrlvl4[j-1, k, i] - 2 * diffpyrlvl4[j, k, i]) * 1.0 / 255          
	                dss = (diffpyrlvl4[j, k, i+1] + diffpyrlvl4[j, k, i-1] - 2 * diffpyrlvl4[j, k, i]) * 1.0 / 255
	                dxy = (diffpyrlvl4[j+1, k+1, i] - diffpyrlvl4[j+1, k-1, i] - diffpyrlvl4[j-1, k+1, i] + diffpyrlvl4[j-1, k-1, i]) * 0.25 / 255 
	                dxs = (diffpyrlvl4[j, k+1, i+1] - diffpyrlvl4[j, k-1, i+1] - diffpyrlvl4[j, k+1, i-1] + diffpyrlvl4[j, k-1, i-1]) * 0.25 / 255 
	                dys = (diffpyrlvl4[j+1, k, i+1] - diffpyrlvl4[j-1, k, i+1] - diffpyrlvl4[j+1, k, i-1] + diffpyrlvl4[j-1, k, i-1]) * 0.25 / 255  

	                dD = np.matrix([[dx], [dy], [ds]])
	                H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
	                x_hat = numpy.linalg.lstsq(H, dD)[0]
	                D_x_hat = diffpyrlvl4[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

	                r = 10.0
	                if (((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(x_hat[0]) < 0.5 and np.absolute(x_hat[1]) < 0.5 and np.absolute(x_hat[2]) < 0.5 and np.absolute(D_x_hat) > 0.03:
	                    extrpyrlvl4[j, k, i - 1] = 1
	                    
	                    
	                    
	print "Fifth octave"

	for i in range(1, 6):
	    for j in range(5, eighted.shape[0] - 5):
	        for k in range(5, eighted.shape[1] - 5):
	            if np.absolute(diffpyrlvl5[j, k, i]) < threshold:
	                continue

	            maxbool = (diffpyrlvl5[j, k, i] > 0)
	            minbool = (diffpyrlvl5[j, k, i] < 0)

	            for di in range(-1, 2):
	                for dj in range(-1, 2):
	                    for dk in range(-1, 2):
	                        if di == 0 and dj == 0 and dk == 0:
	                            continue
	                        maxbool = maxbool and (diffpyrlvl5[j, k, i] > diffpyrlvl5[j + dj, k + dk, i + di])
	                        minbool = minbool and (diffpyrlvl5[j, k, i] < diffpyrlvl5[j + dj, k + dk, i + di])
	                        if not maxbool and not minbool:
	                            break

	                    if not maxbool and not minbool:
	                        break

	                if not maxbool and not minbool:
	                    break

	            if maxbool or minbool:
	                dx = (diffpyrlvl5[j, k+1, i] - diffpyrlvl5[j, k-1, i]) * 0.5 / 255
	                dy = (diffpyrlvl5[j+1, k, i] - diffpyrlvl5[j-1, k, i]) * 0.5 / 255
	                ds = (diffpyrlvl5[j, k, i+1] - diffpyrlvl5[j, k, i-1]) * 0.5 / 255
	                dxx = (diffpyrlvl5[j, k+1, i] + diffpyrlvl5[j, k-1, i] - 2 * diffpyrlvl5[j, k, i]) * 1.0 / 255        
	                dyy = (diffpyrlvl5[j+1, k, i] + diffpyrlvl5[j-1, k, i] - 2 * diffpyrlvl5[j, k, i]) * 1.0 / 255          
	                dss = (diffpyrlvl5[j, k, i+1] + diffpyrlvl5[j, k, i-1] - 2 * diffpyrlvl5[j, k, i]) * 1.0 / 255
	                dxy = (diffpyrlvl5[j+1, k+1, i] - diffpyrlvl5[j+1, k-1, i] - diffpyrlvl5[j-1, k+1, i] + diffpyrlvl5[j-1, k-1, i]) * 0.25 / 255 
	                dxs = (diffpyrlvl5[j, k+1, i+1] - diffpyrlvl5[j, k-1, i+1] - diffpyrlvl5[j, k+1, i-1] + diffpyrlvl5[j, k-1, i-1]) * 0.25 / 255 
	                dys = (diffpyrlvl5[j+1, k, i+1] - diffpyrlvl5[j-1, k, i+1] - diffpyrlvl5[j+1, k, i-1] + diffpyrlvl5[j-1, k, i-1]) * 0.25 / 255  

	                dD = np.matrix([[dx], [dy], [ds]])
	                H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
	                x_hat = numpy.linalg.lstsq(H, dD)[0]
	                D_x_hat = diffpyrlvl5[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)

	                r = 10.0
	                if (((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)) and np.absolute(x_hat[0]) < 0.5 and np.absolute(x_hat[1]) < 0.5 and np.absolute(x_hat[2]) < 0.5 and np.absolute(D_x_hat) > 0.03:
	                    extrpyrlvl5[j, k, i - 1] = 1
	                    
	print "Number of extrema in first octave: %d" % np.sum(extrpyrlvl1)
	print "Number of extrema in second octave: %d" % np.sum(extrpyrlvl2)
	print "Number of extrema in third octave: %d" % np.sum(extrpyrlvl3)
	print "Number of extrema in fourth octave: %d" % np.sum(extrpyrlvl4)
	print "Number of extrema in fifth octave: %d" % np.sum(extrpyrlvl5)

	# Gradient magnitude and orientation for each image sample point at each scale
	magpyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
	magpyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 5))
	magpyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 5))
	magpyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 5))
	magpyrlvl5 = np.zeros((eighted.shape[0], eighted.shape[1], 5))

	oripyrlvl1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
	oripyrlvl2 = np.zeros((normal.shape[0], normal.shape[1], 5))
	oripyrlvl3 = np.zeros((halved.shape[0], halved.shape[1], 5))
	oripyrlvl4 = np.zeros((quartered.shape[0], quartered.shape[1], 5))
	oripyrlvl5 = np.zeros((eighted.shape[0], eighted.shape[1], 5))




	for i in range(0, 5):
	    for j in range(1, doubled.shape[0] - 1):
	        for k in range(1, doubled.shape[1] - 1):
	            magpyrlvl1[j, k, i] = ( ((doubled[j+1, k] - doubled[j-1, k]) ** 2) + ((doubled[j, k+1] - doubled[j, k-1]) ** 2) ) ** 0.5   
	            oripyrlvl1[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((doubled[j, k+1] - doubled[j, k-1]), (doubled[j+1, k] - doubled[j-1, k])))        

	for i in range(0, 5):
	    for j in range(1, normal.shape[0] - 1):
	        for k in range(1, normal.shape[1] - 1):
	            magpyrlvl2[j, k, i] = ( ((normal[j+1, k] - normal[j-1, k]) ** 2) + ((normal[j, k+1] - normal[j, k-1]) ** 2) ) ** 0.5   
	            oripyrlvl2[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((normal[j, k+1] - normal[j, k-1]), (normal[j+1, k] - normal[j-1, k])))    

	for i in range(0, 5):
	    for j in range(1, halved.shape[0] - 1):
	        for k in range(1, halved.shape[1] - 1):
	            magpyrlvl3[j, k, i] = ( ((halved[j+1, k] - halved[j-1, k]) ** 2) + ((halved[j, k+1] - halved[j, k-1]) ** 2) ) ** 0.5   
	            oripyrlvl3[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((halved[j, k+1] - halved[j, k-1]), (halved[j+1, k] - halved[j-1, k])))    

	for i in range(0, 5):
	    for j in range(1, quartered.shape[0] - 1):
	        for k in range(1, quartered.shape[1] - 1):
	            magpyrlvl4[j, k, i] = ( ((quartered[j+1, k] - quartered[j-1, k]) ** 2) + ((quartered[j, k+1] - quartered[j, k-1]) ** 2) ) ** 0.5   
	            oripyrlvl4[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((quartered[j, k+1] - quartered[j, k-1]), (quartered[j+1, k] - quartered[j-1, k])))


	for i in range(0, 5):
	    for j in range(1, eighted.shape[0] - 1):
	        for k in range(1, eighted.shape[1] - 1):
	            magpyrlvl5[j, k, i] = ( ((eighted[j+1, k] - eighted[j-1, k]) ** 2) + ((eighted[j, k+1] - eighted[j, k-1]) ** 2) ) ** 0.5   
	            oripyrlvl5[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((eighted[j, k+1] - eighted[j, k-1]), (eighted[j+1, k] - eighted[j-1, k])))
	            
	            
	            
	extr_sum = np.sum(extrpyrlvl1) + np.sum(extrpyrlvl2) + np.sum(extrpyrlvl3) + np.sum(extrpyrlvl4) + np.sum(extrpyrlvl5)
	extr_sum = int(extr_sum)
	keypoints = np.zeros((extr_sum, 4)) 

	print "Calculating keypoint orientations..."


	count = 0

	for i in range(0, 5):
	    for j in range(80, doubled.shape[0] - 80):
	        for k in range(80, doubled.shape[1] - 80):
	            if extrpyrlvl1[j, k, i] == 1:
	                gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * kvectotal[i]) ** 2))
	                two_sd = np.floor(2 * 1.5 * kvectotal[i])
	                orient_hist = np.zeros([36,1])
	                for x in range(int(-1 * two_sd * 2), int(two_sd * 2) + 1):
	                    ylim = int((((two_sd * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
	                    for y in range(-1 * ylim, ylim + 1):
	                        if j + x < 0 or j + x > doubled.shape[0] - 1 or k + y < 0 or k + y > doubled.shape[1] - 1:
	                            continue
	                        weight = magpyrlvl1[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
	                        bin_idx = np.clip(np.floor(oripyrlvl1[j + x, k + y, i]), 0, 35)
	                        bin_idx = int(np.floor(bin_idx))
	                        orient_hist[bin_idx] += weight  

	                maxval = np.amax(orient_hist)
	                maxidx = np.argmax(orient_hist)
	                keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), kvectotal[i], maxidx])
	                count += 1
	                orient_hist[maxidx] = 0
	                newmaxval = np.amax(orient_hist)
	                while newmaxval > 0.8 * maxval:
	                    newmaxidx = np.argmax(orient_hist)
	                    np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), kvectotal[i], newmaxidx]]), axis=0)
	                    orient_hist[newmaxidx] = 0
	                    newmaxval = np.amax(orient_hist)
	                    
	                    
	                    
	for i in range(0, 5):
	    for j in range(40, normal.shape[0] - 40):
	        for k in range(40, normal.shape[1] - 40):
	            if extrpyrlvl2[j, k, i] == 1:
	                gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * kvectotal[i + 5]) ** 2))
	                two_sd = np.floor(2 * 1.5 * kvectotal[i + 3])
	                orient_hist = np.zeros([36,1])
	                for x in range(int(-1 * two_sd), int(two_sd + 1)):
	                    ylim = int(((two_sd ** 2) - (np.absolute(x) ** 2)) ** 0.5)
	                    for y in range(-1 * ylim, ylim + 1):
	                        if j + x < 0 or j + x > normal.shape[0] - 1 or k + y < 0 or k + y > normal.shape[1] - 1:
	                            continue
	                        weight = magpyrlvl2[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
	                        bin_idx = np.clip(np.floor(oripyrlvl2[j + x, k + y, i]), 0, 35)
	                        bin_idx = int(np.floor(bin_idx))
	                        orient_hist[bin_idx] += weight  

	                maxval = np.amax(orient_hist)
	                maxidx = np.argmax(orient_hist)
	                keypoints[count, :] = np.array([j, k, kvectotal[i + 3], maxidx])
	                count += 1
	                orient_hist[maxidx] = 0
	                newmaxval = np.amax(orient_hist)
	                while newmaxval > 0.8 * maxval:
	                    newmaxidx = np.argmax(orient_hist)
	                    np.append(keypoints, np.array([[j, k, kvectotal[i + 3], newmaxidx]]), axis=0)
	                    orient_hist[newmaxidx] = 0
	                    newmaxval = np.amax(orient_hist)
	                    
	                    
	                    
	for i in range(0, 5):
	    for j in range(20, halved.shape[0] - 20):
	        for k in range(20, halved.shape[1] - 20):
	            if extrpyrlvl3[j, k, i] == 1:
	                gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * kvectotal[i + 10]) ** 2))
	                two_sd = np.floor(2 * 1.5 * kvectotal[i + 6])
	                orient_hist = np.zeros([36,1])
	                for x in range(int(-1 * two_sd * 0.5), int(two_sd * 0.5) + 1):
	                    ylim = int((((two_sd * 0.5) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
	                    for y in range(-1 * ylim, ylim + 1):
	                        if j + x < 0 or j + x > halved.shape[0] - 1 or k + y < 0 or k + y > halved.shape[1] - 1:
	                            continue
	                        weight = magpyrlvl3[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
	                        bin_idx = np.clip(np.floor(oripyrlvl3[j + x, k + y, i]), 0, 35)
	                        bin_idx = int(np.floor(bin_idx))
	                        orient_hist[bin_idx] += weight  

	                maxval = np.amax(orient_hist)
	                maxidx = np.argmax(orient_hist)
	                keypoints[count, :] = np.array([j * 2, k * 2, kvectotal[i + 6], maxidx])
	                count += 1
	                orient_hist[maxidx] = 0
	                newmaxval = np.amax(orient_hist)
	                while newmaxval > 0.8 * maxval:
	                    newmaxidx = np.argmax(orient_hist)
	                    np.append(keypoints, np.array([[j * 2, k * 2, kvectotal[i + 6], newmaxidx]]), axis=0)
	                    orient_hist[newmaxidx] = 0
	                    newmaxval = np.amax(orient_hist)
	                    
	                
	for i in range(0, 5):
	        for j in range(10, quartered.shape[0] - 10):
	            for k in range(10, quartered.shape[1] - 10):
	                if extrpyrlvl4[j, k, i] == 1:
	                    gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * kvectotal[i + 15]) ** 2))
	                    two_sd = np.floor(2 * 1.5 * kvectotal[i + 9])
	                    orient_hist = np.zeros([36,1])
	                    for x in range(int(-1 * two_sd * 0.25), int(two_sd * 0.25) + 1):
	                        ylim = int((((two_sd * 0.25) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
	                        for y in range(-1 * ylim, ylim + 1):
	                            if j + x < 0 or j + x > quartered.shape[0] - 1 or k + y < 0 or k + y > quartered.shape[1] - 1:
	                                continue
	                            weight = magpyrlvl4[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
	                            bin_idx = np.clip(np.floor(oripyrlvl4[j + x, k + y, i]), 0, 35)
	                            bin_idx = int(np.floor(bin_idx))
	                            orient_hist[bin_idx] += weight  

	                    maxval = np.amax(orient_hist)
	                    maxidx = np.argmax(orient_hist)
	                    keypoints[count, :] = np.array([j * 4, k * 4, kvectotal[i + 9], maxidx])
	                    count += 1
	                    orient_hist[maxidx] = 0
	                    newmaxval = np.amax(orient_hist)
	                    while newmaxval > 0.8 * maxval:
	                        newmaxidx = np.argmax(orient_hist)
	                        np.append(keypoints, np.array([[j * 4, k * 4, kvectotal[i + 9], newmaxidx]]), axis=0)
	                        orient_hist[newmaxidx] = 0
	                        newmaxval = np.amax(orient_hist)
	                        
	                        
	for i in range(0, 5):
	    for j in range(5, eighted.shape[0] - 5):
	        for k in range(5, eighted.shape[1] - 5):
	            if extrpyrlvl5[j, k, i] == 1:
	                gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * kvectotal[i + 20]) ** 2))
	                two_sd = np.floor(2 * 1.5 * kvectotal[i + 12])
	                orient_hist = np.zeros([36,1])
	                for x in range(int(-1 * two_sd * 0.25), int(two_sd * 0.25) + 1):
	                    ylim = int((((two_sd * 0.25) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
	                    for y in range(-1 * ylim, ylim + 1):
	                        if j + x < 0 or j + x > eighted.shape[0] - 1 or k + y < 0 or k + y > eighted.shape[1] - 1:
	                            continue
	                        weight = magpyrlvl5[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
	                        bin_idx = np.clip(np.floor(oripyrlvl5[j + x, k + y, i]), 0, 35)
	                        bin_idx = int(np.floor(bin_idx))
	                        orient_hist[bin_idx] += weight  
	                
	                maxval = np.amax(orient_hist)
	                maxidx = np.argmax(orient_hist)
	                keypoints[count, :] = np.array([j * 4, k * 4, kvectotal[i + 12], maxidx])
	                count += 1
	                orient_hist[maxidx] = 0
	                newmaxval = np.amax(orient_hist)
	                while newmaxval > 0.8 * maxval:
	                    newmaxidx = np.argmax(orient_hist)
	                    np.append(keypoints, np.array([[j * 4, k * 4, kvectotal[i + 12], newmaxidx]]), axis=0)
	                    orient_hist[newmaxidx] = 0
	                    newmaxval = np.amax(orient_hist)
	                    
	                    
	                    
	                    
	print "Calculating descriptor..."

	magpyr = np.zeros((normal.shape[0], normal.shape[1], 25))
	oripyr = np.zeros((normal.shape[0], normal.shape[1], 25))

	for i in range(0, 5):
	    magmax = np.amax(magpyrlvl1[:, :, i])
	    magpyr[:, :, i] = misc.imresize(magpyrlvl1[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(float)
	    magpyr[:, :, i] = (magmax / np.amax(magpyr[:, :, i])) * magpyr[:, :, i]  
	    oripyr[:, :, i] = misc.imresize(oripyrlvl1[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int)    
	    oripyr[:, :, i] = ((36.0 / np.amax(oripyr[:, :, i])) * oripyr[:, :, i]).astype(int)

	for i in range(0, 5):
	    magpyr[:, :, i+5] = (magpyrlvl2[:, :, i]).astype(float)
	    oripyr[:, :, i+5] = (oripyrlvl2[:, :, i]).astype(int)             

	for i in range(0, 5):
	    magpyr[:, :, i+10] = misc.imresize(magpyrlvl3[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int)   
	    oripyr[:, :, i+10] = misc.imresize(oripyrlvl3[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int)    

	for i in range(0, 5):
	    magpyr[:, :, i+15] = misc.imresize(magpyrlvl4[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int)   
	    oripyr[:, :, i+15] = misc.imresize(oripyrlvl4[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int)

	for i in range(0, 5):
	    magpyr[:, :, i+20] = misc.imresize(magpyrlvl5[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int)   
	    oripyr[:, :, i+20] = misc.imresize(oripyrlvl5[:, :, i], (normal.shape[0], normal.shape[1]), "bilinear").astype(int) 


	descriptors = np.zeros([keypoints.shape[0], 128])

	for i in range(0, keypoints.shape[0]): 
	    for x in range(-8, 8):
	        for y in range(-8, 8):
	            theta = 10 * keypoints[i,3] * np.pi / 180.0
	            xrot = np.round((np.cos(theta) * x) - (np.sin(theta) * y))
	            yrot = np.round((np.sin(theta) * x) + (np.cos(theta) * y))
	            xrot = int(xrot)
	            yrot = int(yrot)
	            scale_idx = np.argwhere(kvectotal == keypoints[i,2])[0][0]
	            scale_idx = int(scale_idx)
	            x0 = keypoints[i,0]
	            y0 = keypoints[i,1]
	            y0 = int(y0)
	            x0 = int(x0)
	            gaussian_window = multivariate_normal(mean=[x0,y0], cov=8) 
	            weight = magpyr[x0 + xrot, y0 + yrot, scale_idx] * gaussian_window.pdf([x0 + xrot, y0 + yrot])
	            angle = oripyr[x0 + xrot, y0 + yrot, scale_idx] - keypoints[i,3]
	            if angle < 0:
	                angle = 36 + angle

	            bin_idx = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
	            descriptors[i, 32 * int((x + 8)/4) + 8 * int((y + 8)/4) + bin_idx] += weight

	    descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :]) 
	    descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)
	    descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
	    
	    return [keypoints, descriptors]





def match_keypoints(imagename, templatename):

	threshold = 5
	cutoff = 0.9
	img = ndimage.imread(imagename)
	template = ndimage.imread(templatename)

	[kpi, di] = detect_keypoints(imagename, threshold)
	[kpt, dt] = detect_keypoints(templatename, threshold)

	flann_params = dict(algorithm=1, trees=4)
	flann = cv2.flann_Index(np.asarray(di, np.float32), flann_params)
	idx, dist = flann.knnSearch(np.asarray(dt, np.float32), 1, params={})
	del flann

	dist = dist[:,0]/2500.0
	dist = dist.reshape(-1,).tolist()
	idx = idx.reshape(-1).tolist()
	indices = range(len(dist))
	indices.sort(key=lambda i: dist[i])
	dist = [dist[i] for i in indices]
	idx = [idx[i] for i in indices]

	kpi_cut = []
	for i, dis in itertools.izip(idx, dist):
	    if dis < cutoff:
	        kpi_cut.append(kpi[i])
	    else:
	        break

	kpt_cut = []
	for i, dis in itertools.izip(indices, dist):
	    if dis < cutoff:
	        kpt_cut.append(kpt[i])
	    else:
	        break

	h1, w1 = img.shape[:2]
	h2, w2 = template.shape[:2]
	nWidth = w1 + w2
	nHeight = max(h1, h2)
	hdif = (h1 - h2) / 2
	newimg = np.zeros((nHeight, nWidth), np.uint8)
	newimg[hdif:hdif+h2, :w2] = template
	newimg[:h1, w2:w1+w2] = img

	for i in range(min(len(kpi), len(kpt))):
	    pt_a = (int(kpt[i,1]), int(kpt[i,0] + hdif))
	    pt_b = (int(kpi[i,1] + w2), int(kpi[i,0]))
	    cv2.line(newimg, pt_a, pt_b, (255, 0, 0))

	cv2.imwrite('matches.jpg', newimg)




def main():
    imagename = sys.argv[1]
    templatename = sys.argv[2]

    match_keypoints(imagename, templatename)

if __name__ == '__main__':
    main()