# USAGE
# python classify-all.py --model filter_v2.model --labelbin label_v2.pickle --images data

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
#import tensorflow as tf
import argparse
import pickle
from PIL import Image
import os

""" # Configure amount of GPU memory in % to allocate:
config = tf.GPUOptions(per_process_gpu_memory_fraction=0.870)
sess = tf.Session(config=tf.ConfigProto(gpu_options=config)) """

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--images", required=True,
	help="path to input images")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# Create output directory and subdirectories
if not os.path.exists('output'):
	os.mkdir('output')
	for cl in lb.classes_:
		os.makedirs('output/'+cl)

# loop over all images in each subdirectory
print("[INFO] classifying images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
for imagePath in imagePaths:
	# load and save a copy of the image
	image = Image.open(imagePath)
	output = image.copy()
	ppn = imagePath.split(os.path.sep)[-2]
	filename = imagePath.split(os.path.sep)[-1]

	# pre-process the image for classification
	image = image.resize((96, 96))
	image = np.array(image)
	image = np.divide(image, 255.0)
	image = np.expand_dims(image, axis=0)

	# classify the input image
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	# within the predicted subdirectory of the class create
	# another subdirectory according to the PPN of the image.
	out_path = 'output/'+label+'/'+ppn
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	output.save(os.path.join(out_path,filename))

print("[INFO] Done! Output directory created!")
