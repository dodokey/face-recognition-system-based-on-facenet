import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
from recognition import facenet

image_size = 160 #don't need equal to real image size, but this value should not small than this
modeldir = 'recognition/model/20180402-114759.pb' #change to your model dir
image_name1 = 'media/test3.JPG' #change to your image name
image_name2 = 'media/test4.JPG' #change to your image name
image_name3 = 'media/test1.JPG' #change to your image name

print('facenet embedding')
tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

print('facenet embedding')

scaled_reshape = []

image1 = scipy.misc.imread(image_name1, mode='RGB')
image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image1 = facenet.prewhiten(image1)
scaled_reshape.append(image1.reshape(-1,image_size,image_size,3))
emb_array1 = np.zeros((1, embedding_size))
emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]


image2 = scipy.misc.imread(image_name2, mode='RGB')
image2 = cv2.resize(image2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image2 = facenet.prewhiten(image2)
scaled_reshape.append(image2.reshape(-1,image_size,image_size,3))
emb_array2 = np.zeros((1, embedding_size))
emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]


image3 = scipy.misc.imread(image_name3, mode='RGB')
image3 = cv2.resize(image3, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image3 = facenet.prewhiten(image3)
scaled_reshape.append(image3.reshape(-1,image_size,image_size,3))
emb_array3 = np.zeros((1, embedding_size))
emb_array3[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]


dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
dist2 = np.sqrt(np.sum(np.square(emb_array3[0]-emb_array2[0])))
print("dist of same face: %f "%dist)
print("dist of diff face: %f "%dist2)