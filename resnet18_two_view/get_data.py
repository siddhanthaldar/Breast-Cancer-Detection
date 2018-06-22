import convert
import os
import cv2
import numpy as np

ROOT_DIR = os.path.abspath("../")
path_to_ddsm = os.path.join(ROOT_DIR,"data/")


def get_data(path_to_ddsm):
	cc = np.zeros((18,224,224))
	mlo = np.zeros((18,224,224))
	labels = np.zeros((18,3))

	if os.path.isfile("cc.npy") and os.path.isfile("mlo.npy") and os.path.isfile("labels.npy"):
		print("Loading train files from memory")
		cc = np.load("cc.npy")
		mlo = np.load("mlo.npy")
		labels = np.load("labels.npy")
		return cc, mlo, labels

	benign_right_cc = []
	benign_left_cc = []
	benign_right_mlo = []
	benign_left_mlo = []

	normal_right_cc = []
	normal_left_cc = []
	normal_right_mlo = []
	normal_left_mlo = []
	
	cancer_right_cc = []
	cancer_left_cc = []
	cancer_right_mlo = []
	cancer_left_mlo = []

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
					if "LEFT" in file_name:
						if "CC" in file_name:
							cancer_left_cc.append(image)
						elif "MLO" in file_name:
							cancer_left_mlo.append(image)

					elif "RIGHT" in file_name:
						if "CC" in file_name:
							cancer_right_cc.append(image)
						elif "MLO" in file_name:
							cancer_right_mlo.append(image)		

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
					if "LEFT" in file_name:
						if "CC" in file_name:
							benign_left_cc.append(image)
						elif "MLO" in file_name:
							benign_left_mlo.append(image)

					elif "RIGHT" in file_name:
						if "CC" in file_name:
							benign_right_cc.append(image)
						elif "MLO" in file_name:
							benign_right_mlo.append(image)

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
					if "LEFT" in file_name:
						if "CC" in file_name:
							normal_left_cc.append(image)
						elif "MLO" in file_name:
							normal_left_mlo.append(image)

					elif "RIGHT" in file_name:
						if "CC" in file_name:
							normal_right_cc.append(image)
						elif "MLO" in file_name:
							normal_right_mlo.append(image)

	cancer_cc = cancer_left_cc + cancer_right_cc
	cancer_mlo = cancer_left_mlo + cancer_right_mlo

	benign_cc = benign_left_cc + benign_right_cc
	benign_mlo = benign_left_mlo + benign_right_mlo

	normal_cc = normal_left_cc + normal_right_cc
	normal_mlo = normal_left_mlo + normal_right_mlo

	#Labels
	labels = np.zeros((len(cancer_cc) + len(benign_cc) + len(normal_cc),3))
	# 0 - normal, 1 - benign, 2 - cancer
	for i in range(len(cancer_cc)):
		labels[i][2] = 1
	for i in range(len(benign_cc)):
		labels[i + len(cancer_cc)][1] = 1
	for i in range(len(normal_cc)):
		labels[i + len(cancer_cc) + len(benign_cc)][0] = 1  

	cc = cancer_cc + benign_cc + normal_cc
	mlo = cancer_mlo + benign_mlo + normal_mlo

	cc_array = np.asarray(cc)
	mlo_array = np.asarray(mlo)	  

	np.save("cc", cc_array)
	np.save("mlo", mlo_array)
	np.save("labels", labels)

	file = open("array_dim.txt","w")
	file.write(str(cc_array.shape))
	file.write(str(mlo_array.shape))
	file.write(str(labels.shape))
	file.close()

	return cc_array, mlo_array, labels				



if __name__=="__main__":
	get_data(path_to_ddsm)