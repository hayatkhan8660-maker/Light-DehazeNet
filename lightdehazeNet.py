# @author: hayat
import torch
import torch.nn as nn
import math

class LightDehaze_Net(nn.Module):

	def __init__(self):
		super(LightDehaze_Net, self).__init__()
		
		# LightDehazeNet Architecture 
		self.relu = nn.ReLU(inplace=True)

		self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
		self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
		self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 
		self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True) 
		self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
		self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
		self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
		self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)
		
	def forward(self, img):
		pipeline = []
		pipeline.append(img)

		conv_layer1 = self.relu(self.e_conv_layer1(img))
		conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
		conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

		# concatenating conv1 and conv3
		concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)
		
		conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
		conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
		conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

		# concatenating conv4 and conv6
		concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)
		
		conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

		# concatenating conv2, conv5, and conv7
		concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)
		
		conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


		dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
		#J(x) = clean_image, k(x) = x8, I(x) = x, b = 1
		
		
		return dehaze_image 

		


			

			
			






