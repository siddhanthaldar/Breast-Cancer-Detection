import convert
import os
import cv2
import numpy as np

ROOT_DIR = os.path.abspath("../")
path_to_ddsm = os.path.join(ROOT_DIR,"data/")


def get_data(path_to_ddsm):
	cancer_train = np.zeros((12,224,224))
	benign_train = np.zeros((8,224,224))
	normal_train = np.zeros((16,224,224))

	if os.path.isfile("cancer.npy") and os.path.isfile("benign.npy") and os.path.isfile("normal.npy"):
		print("Loading train files from memory")
		cancer_train = np.load("cancer.npy")
		benign_train = np.load("benign.npy")
		normal_train = np.load("normal.npy")
		return cancer_train, benign_train, normal_train

	benign_img = []
	normal_img = []
	cancer_img = []

	cancer_path = os.path.join(path_to_ddsm,"cancers/")
	benign_path = os.path.join(path_to_ddsm,"benigns/")
	normal_path = os.path.join(path_to_ddsm,"normal/")

	count = 0

	image_shape = (224,224)

	for root, subFolders, file_names in os.walk(cancer_path):
		for file_name in file_names:
			if ".LJPEG" in file_name:
				ljpeg_path = os.path.join(root, file_name)
				out_path = os.path.join(root, file_name)
				out_path = out_path.split('.LJPEG')[0] + ".jpg"
				image = convert.ljpeg_to_jpg(ljpeg = ljpeg_path, output = out_path, scale = 0.10)
				if image is not None:
					count += 1
					image = cv2.resize(image, image_shape)
					cancer_img.append(image)

	# print(count)

	for root, subFolders, file_names in os.walk(benign_path):
		for file_name in file_names:
			if ".LJPEG" in file_name:
				ljpeg_path = os.path.join(root, file_name)
				out_path = os.path.join(root, file_name)
				out_path = out_path.split('.LJPEG')[0] + ".jpg"
				image = convert.ljpeg_to_jpg(ljpeg = ljpeg_path, output = out_path, scale = 0.10)
				if image is not None:
					count += 1
					image = cv2.resize(image, image_shape)
					benign_img.append(image)

	for root, subFolders, file_names in os.walk(normal_path):
		for file_name in file_names:
			if ".LJPEG" in file_name:
				ljpeg_path = os.path.join(root, file_name)
				out_path = os.path.join(root, file_name)
				out_path = out_path.split('.LJPEG')[0] + ".jpg"
				image = convert.ljpeg_to_jpg(ljpeg = ljpeg_path, output = out_path, scale = 0.10)
				if image is not None:
					count += 1
					image = cv2.resize(image, image_shape)
					normal_img.append(image)

	cancer = np.asarray(cancer_img)
	benign = np.asarray(benign_img)
	normal = np.asarray(normal_img)

	np.save("cancer", cancer)
	np.save("benign", benign)
	np.save("normal", normal)

	return cancer,benign,normal				



if __name__=="__main__":
	get_data(path_to_ddsm)