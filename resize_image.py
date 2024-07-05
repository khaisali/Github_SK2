# 'resizing the image using opencv'Not part of project
import cv2
def re(path):
	image = cv2.imread(path, 1)
	# Loading the image
	final_pic = cv2.resize(image, (1080, 720),
				interpolation = cv2.INTER_LINEAR)
	# Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
	images =[ final_pic]
	count = 1
	return final_pic

path=r"C:\Users\dell\Desktop\mini\Minor-2\ImageData\anger\S010_004_00000019.png"

#re(image)
image = cv2.imread(path, 1)
half = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
bigger = cv2.resize(image, (1050, 1610))

stretch_near = cv2.resize(image, (780, 540),
						  interpolation=cv2.INTER_LINEAR)

Titles = ["Original", "Half", "Bigger", "Interpolation Nearest"]
images = [image, half, bigger, stretch_near]
count = 4

import matplotlib.pyplot as plt

for i in range(count):
	plt.subplot(2, 2, i + 1)
	plt.title(Titles[i])
	plt.imshow(images[i])

plt.show()