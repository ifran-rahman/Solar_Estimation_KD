# Import necessary libraries 
import torch 
from PIL import Image 
import torchvision.transforms as transforms 

def imagetoTensor(img_path):
	# Read a PIL image 
	image = Image.open(img_path) 

	# Define a transform to convert PIL 
	# image to a Torch tensor 
	transform = transforms.Compose([ 
		transforms.PILToTensor() 
	]) 

	# transform = transforms.PILToTensor() 
	# Convert the PIL image to Torch tensor 
	img_tensor = transform(image) 

	# print the converted Torch tensor 
	return img_tensor
