# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np
import urllib
import pylab
import json
import cv2
import os

@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
			# load the image and convert
			image = _grab_image(url=url)
		### START WRAPPING OF COMPUTER VISION APP
		# Insert code here to process the image and update
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('gray_image.png',gray_image)
		cv2.getGaussianKernel(9,9)
		blur= cv2.GaussianBlur(image,(5,5),0)
		cv2.imwrite('blur.png',blur)
		image=cv2.imread('blur.png')
		kernel=np.ones((5,5),np.float32)/25
		dst= cv2.filter2D(image,-1,kernel)
		plt.subplot(121)
		plt.xticks([]), plt.yticks([])
		plt.subplot(122)
		plt.xticks([]), plt.yticks([])
		cv2.imwrite('averaged.png',dst)
		image = cv2.imread('averaged.png')
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		cv2.imwrite('thresh_image.jpg',thresh)
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
		sure_bg = cv2.dilate(opening,kernel,iterations=3)
		print ("No. of pests in the image: ")
		labelarray, particle_count = ndimage.measurements.label(sure_bg)
		print (particle_count)
		# the `data` dictionary with your results
		### END WRAPPING OF COMPUTER VISION APP
		# update the data dictionary
		data["success"] = True
	# return a JSON response
	return JsonResponse(data)
def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
	#otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image