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

		self.fc_concat = nn.Linear(1024,3)
		self.fc_add = nn.Linear(512,3)

	def forward(self,x1,x2,mode):
		x1 = self.net1(x1)
		# x1 = self.fc(x1)	
		x2 = self.net1(x2)
		# x2 = self.fc(x2)

		if mode == "concat":
			# print(x1.shape)
			# print(x2.shape)
			x = torch.cat((x1,x2),1)
			x = self.fc_concat(x)
		elif mode == "add":
			x = x1 + x2
			x = self.fc_add(x)
		return x	

# def calc_accuracy(mdl,X,Y):
# 	max_vals, max_indices = torch.max(mdl(X[:,0,:,:].unsqueeze(1),X[:,1,:,:].unsqueeze(1), "concat"),1)
# 	train_acc = (max_indices == torch.max(Y,1)[1]).sum().data.numpy()/max_indices.size()[0]
# 	return train_acc

def train():
	cc_imgs, mlo_imgs, labels = get_data.get_data(path_to_ddsm)

	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("GPU is available")

	use_gpu = False	

	# In PyTorch, images are (channels, height, width)

	#Make Tensors
	cc_tensors = torch.FloatTensor(cc_imgs.shape[0], 1,cc_imgs.shape[1], cc_imgs.shape[2] )
	mlo_tensors = torch.FloatTensor(mlo_imgs.shape[0], 1,mlo_imgs.shape[1], mlo_imgs.shape[2] )
	labels_tensors = torch.LongTensor(labels.shape[0], labels.shape[1])
	TrainImages = torch.FloatTensor(cc_imgs.shape[0], 2,cc_imgs.shape[1], cc_imgs.shape[2] )

	for i,img in enumerate(cc_imgs):
		cc_tensors[i] = torch.FloatTensor(img).unsqueeze(0)

	for i,img in enumerate(mlo_imgs):
		mlo_tensors[i] = torch.FloatTensor(img).unsqueeze(0)
			
	for i, label in enumerate(labels):
		labels_tensors[i] = torch.LongTensor(label)	

	# print(cc_tensors.shape)

	TrainImages = torch.cat((cc_tensors, mlo_tensors), 1)	
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
	optimizer = optim.Adam(net.parameters(), lr=1e-4)

	iterations = 50
	train_loss = []
	num_samples = cc_imgs.shape[0] #number of training samples

	#Path for saving weights
	filepath = os.path.join(ROOT_DIR,"resnet18_two_view/weight.pt")
	# #Load weights
	# net = torch.load(filepath)

	for epoch in range(iterations):
		runningLoss = 0
		net.train(True)
		for data in trainLoader:
			imgs, labels = data
			# print(imgs[:,0,:,:].unsqueeze(1).shape)
			# print(imgs[:,1,:,:].unsqueeze(1).shape)
			cc_img = imgs[:,0,:,:].unsqueeze(1)
			mlo_img = imgs[:,1,:,:].unsqueeze(1)

			if use_gpu:
				imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
			else:	 	
				imgs, labels = Variable(imgs), Variable(labels)

			outputs = net(imgs[:,0,:,:].unsqueeze(1),imgs[:,1,:,:].unsqueeze(1), "concat")

			loss = criterion(outputs, torch.max(labels, 1)[1])
			optimizer.zero_grad()
			loss.backward()
			#Update network params
			optimizer.step()

			#Accumulate loss per batch
			runningLoss += loss.data[0]
		avgTrainLoss = runningLoss/num_samples
		train_loss.append(avgTrainLoss)	

		#Calculate batch accuracy
		# accuracy = calc_accuracy(net,imgs,labels)

		# Plotting Loss vs Epochs
		fig1 = plt.figure(1)
		plt.plot(range(epoch+1), train_loss, 'r--', label = 'train')
		if epoch == 0:
			plt.legend(loc='upper left')
			plt.xlabel('Epochs')
			plt.ylabel('Loss')

		print('At Iteration:'+str(epoch+1)+'   Training Loss : ' + str(avgTrainLoss))#+'    Accuracy : ' + str(accuracy))	
	
		if epoch % 5 == 0:
			torch.save(net.state_dict(), filepath)

	plt.show()	

if __name__ == "__main__":
	# train_imgs, train_labels = load_data()	
	# print(train_imgs.shape)
	# print(train_labels.shape)

	train()