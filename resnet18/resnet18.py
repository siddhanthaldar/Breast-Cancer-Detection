import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
import get_data
import os
import matplotlib.pyplot as plt 

# Load Data as numpy array
def load_data():
	ROOT_DIR = os.path.abspath("../")
	path_to_ddsm = os.path.join(ROOT_DIR,"data/")

	cancer_imgs, benign_imgs, normal_imgs = get_data.get_data(path_to_ddsm)

	# print(cancer_imgs.shape)
	# print(benign_imgs.shape)
	# print(normal_imgs.shape)

	train_imgs = np.concatenate((cancer_imgs, benign_imgs, normal_imgs), axis = 0)
	train_labels = np.zeros((train_imgs.shape[0],3))  

	# 0 - normal, 1 - benign, 2 - cancer
	for i in range(cancer_imgs.shape[0]):
		train_labels[i][2] = 1
	for i in range(benign_imgs.shape[0]):
		train_labels[i + cancer_imgs.shape[0]][1] = 1
	for i in range(normal_imgs.shape[0]):
		train_labels[i + cancer_imgs.shape[0] + benign_imgs.shape[0]][0] = 1 

	return train_imgs, train_labels

def train():
	train_imgs, train_labels = load_data()

	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("GPU is available")

	use_gpu = False

	#Make Tensors
	TrainImages = torch.FloatTensor(train_imgs.shape[0], 1,train_imgs.shape[1], train_imgs.shape[2] )
	TrainLabels = torch.LongTensor(train_labels.shape[0], train_labels.shape[1])

	for i,img in enumerate(train_imgs):
		TrainImages[i] = torch.FloatTensor(img).unsqueeze(0)
	for i, label in enumerate(train_labels):
		TrainLabels[i] = torch.LongTensor(label)	

	#Create PyTorch dataset
	train_dataset = TensorDataset(TrainImages, TrainLabels)

	#Create DataLoader
	batch_size = 1
	trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)		

	#Model Creation
	net = models.resnet18(pretrained = False)
	# net.fc.out_features = 3
	net.fc = nn.Linear(512,3,bias=True)
	net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
	# print(net)
	if use_gpu:
		net = net.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=1e-3)

	iterations = 100
	train_loss = []
	num_samples = train_imgs.shape[0] #number of training samples

	for epoch in range(iterations):
		runningLoss = 0
		net.train(True)
		for data in trainLoader:
			imgs, labels = data
			if use_gpu:
				imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
			else:	 	
				imgs, labels = Variable(imgs), Variable(labels)

			outputs = net(imgs)
			print(outputs.shape)
			
			loss = criterion(outputs, torch.max(labels, 1)[1])
			optimizer.zero_grad()
			loss.backward()
			#Update network params
			optimizer.step()
			#Accumulate loss per batch
			runningLoss += loss.data[0]
		avgTrainLoss = runningLoss/num_samples
		train_loss.append(avgTrainLoss)	

		# Plotting Loss vs Epochs
		fig1 = plt.figure(1)
		plt.plot(range(epoch+1), train_loss, 'r--', label = 'train')
		if epoch == 0:
			plt.legend(loc='upper left')
			plt.xlabel('Epochs')
			plt.ylabel('Loss')

		print('At Iteration:'+str(epoch+1)+'   Training Loss : ' + str(avgTrainLoss))	
	
	plt.show()	

if __name__ == "__main__":
	# train_imgs, train_labels = load_data()	
	# print(train_imgs.shape)
	# print(train_labels.shape)

	train()