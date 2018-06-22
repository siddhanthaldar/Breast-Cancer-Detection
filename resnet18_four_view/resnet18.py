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

ROOT_DIR = os.path.abspath("../")
path_to_ddsm = os.path.join(ROOT_DIR,"data/")

def resnet_model():
	net = models.resnet18(pretrained=False)
	# net.fc.out_features = 512
	net.fc = nn.Linear(512,512,bias=True)
	net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

	return net


class Resnet18_twoview(torch.nn.Module):

	def __init__(self):
		super(Resnet18_twoview, self).__init__()

		# self.net1 = models.resnet18(pretrained=False)
		# self.net1.fc.out_features = 512
		# self.net1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

		self.net1 = resnet_model()
		# self.fc = nn.Linear(1000,512)

		self.fc_concat = nn.Linear(512*4,3)
		self.fc_add = nn.Linear(512,3)

	def forward(self,x1,x2,x3,x4,mode):
		x1 = self.net1(x1)
		# x1 = self.fc(x1)	
		x2 = self.net1(x2)
		# x2 = self.fc(x2)
		x3 = self.net1(x3)
		# x3 = self.fc(x3)	
		x4 = self.net1(x4)
		# x4 = self.fc(x4)	
		

		if mode == "concat":
			x = torch.cat((x1,x2,x3,x4),1)
			x = self.fc_concat(x)
		elif mode == "add":
			x = x1 + x2 + x3 + x4
			x = self.fc_add(x)
		return x	


def train():
	left_cc_imgs, right_cc_imgs, left_mlo_imgs, right_mlo_imgs, labels = get_data.get_data(path_to_ddsm)

	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("GPU is available")

	use_gpu = False	

	# In PyTorch, images are (channels, height, width)

	#Make Tensors
	left_cc_tensors = torch.FloatTensor(left_cc_imgs.shape[0], 1,left_cc_imgs.shape[1], left_cc_imgs.shape[2] )
	right_cc_tensors = torch.FloatTensor(right_cc_imgs.shape[0], 1,right_cc_imgs.shape[1], right_cc_imgs.shape[2] )
	left_mlo_tensors = torch.FloatTensor(left_mlo_imgs.shape[0], 1,left_mlo_imgs.shape[1], left_mlo_imgs.shape[2] )
	right_mlo_tensors = torch.FloatTensor(right_mlo_imgs.shape[0], 1, right_mlo_imgs.shape[1], right_mlo_imgs.shape[2] )
	labels_tensors = torch.LongTensor(labels.shape[0], labels.shape[1])
	
	TrainImages = torch.FloatTensor(left_cc_imgs.shape[0], 4, left_cc_imgs.shape[1], left_cc_imgs.shape[2] )

	for i,img in enumerate(left_cc_imgs):
		left_cc_tensors[i] = torch.FloatTensor(img).unsqueeze(0)

	for i,img in enumerate(right_cc_imgs):
		right_cc_tensors[i] = torch.FloatTensor(img).unsqueeze(0)

	for i,img in enumerate(left_mlo_imgs):
		left_mlo_tensors[i] = torch.FloatTensor(img).unsqueeze(0)
	
	for i,img in enumerate(right_mlo_imgs):
		right_mlo_tensors[i] = torch.FloatTensor(img).unsqueeze(0)
			
	for i, label in enumerate(labels):
		labels_tensors[i] = torch.LongTensor(label)	

	print(left_cc_tensors.shape)

	TrainImages = torch.cat((left_cc_tensors, right_cc_tensors, left_mlo_tensors, right_mlo_tensors), 1)	
	# print(TrainImages.shape)

	#Create PyTorch dataset
	train_dataset = TensorDataset(TrainImages,labels_tensors)


	#Create DataLoader
	batch_size = 2
	trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)		

	net = Resnet18_twoview()

	# print(net)
	if use_gpu:
		net = net.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=1e-3)

	iterations = 100
	train_loss = []
	num_samples = left_cc_imgs.shape[0] #number of training samples

	for epoch in range(iterations):
		runningLoss = 0
		net.train(True)
		for data in trainLoader:
			imgs, labels = data
			# print(imgs[:,3,:,:].unsqueeze(1).shape)
			# print(imgs[:,1,:,:].unsqueeze(1).shape)
			
			if use_gpu:
				imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
			else:	 	
				imgs, labels = Variable(imgs), Variable(labels)

			left_cc_img = imgs[:,0,:,:].unsqueeze(1)
			right_cc_img = imgs[:,1,:,:].unsqueeze(1)
			left_mlo_img = imgs[:,2,:,:].unsqueeze(1)
			right_mlo_img = imgs[:,3,:,:].unsqueeze(1)

			# print(left_cc_img)
			outputs = net(left_cc_img, right_cc_img, left_mlo_img, right_mlo_img, "concat")
			
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