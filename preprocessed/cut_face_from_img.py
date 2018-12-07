import cv2
import time
import numpy as np
from detection.FaceDetector import FaceDetector
from recognition.FaceRecognition import FaceRecognition

import os
import scipy.misc

face_detector = FaceDetector()
face_recognition = FaceRecognition()

feature_dict = dict()

def cutface(filepath, filename):
	#face_detector = FaceDetector()
	img = scipy.misc.imread(filepath+'/1.jpg', mode='RGB')
	boxes, scores = face_detector.detect(img)

	face_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
	face_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
	boxx= face_boxes[0]
	image1 = img[boxx[0]:boxx[2], boxx[1]:boxx[3], :]
	image1 = cv2.resize(image1, (160, 160), interpolation=cv2.INTER_AREA)
	#cv2.imwrite('./media/check/'+ filename +'.jpg',image1)
	feature = face_recognition.recognize(image1)
	feature_dict[filename] = feature
	#cv2.imshow(filename, image1)

	cv2.waitKey(1)
	return


datadir = './media/train_classifier'
for file in os.listdir(datadir):
    cutface(datadir+'/'+ file, file)
np.save('feature_dict.npy', feature_dict) 





