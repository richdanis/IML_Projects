#imports
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from os import listdir

INF = 1e10
food = '../Data/food/'

def get_ima(i):
    pic_im = Image.open(food + i).convert('RGB')
    to_tensor = transforms.ToTensor()
    pic_ten = to_tensor(pic_im)
    return pic_ten.shape

min_a2 = INF
min_a3 = INF

k = listdir(food)

for i in k:
    if i == '.DS_Store':
        continue
    A,B,C = get_ima(i)
    if B < min_a2:
        min_a2 = B
    if C < min_a3:
        min_a3 = C


print(min_a2) #242
print(min_a3) #354