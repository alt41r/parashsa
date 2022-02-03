import cv2 as cv
import numpy as np
import random as rng
from resnet import make_emotion

import cv2
import numpy as np
import sys
import os
import time
import torch
import torch.nn as nn
import pickle




def conv_block(in_channels, out_channels, pool=False):
	layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			  nn.BatchNorm2d(out_channels),
			  nn.ELU(inplace=True)]
	if pool: layers.append(nn.MaxPool2d(2))
	return nn.Sequential(*layers)


class ResNet(nn.Module):
	def __init__(self, in_channels, num_classes):
		super().__init__()

		self.conv1 = conv_block(in_channels, 128)
		self.conv2 = conv_block(128, 128, pool=True)
		self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
		self.drop1 = nn.Dropout(0.5)

		self.conv3 = conv_block(128, 256)
		self.conv4 = conv_block(256, 256, pool=True)
		self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
		self.drop2 = nn.Dropout(0.5)

		self.conv5 = conv_block(256, 512)
		self.conv6 = conv_block(512, 512, pool=True)
		self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
		self.drop3 = nn.Dropout(0.5)

		self.classifier = nn.Sequential(nn.MaxPool2d(6),
										nn.Flatten(),
										nn.Linear(512, num_classes))

	def forward(self, xb):
		out = self.conv1(xb)
		out = self.conv2(out)
		out = self.res1(out) + out
		out = self.drop1(out)

		out = self.conv3(out)
		out = self.conv4(out)
		out = self.res2(out) + out
		out = self.drop2(out)

		out = self.conv5(out)
		out = self.conv6(out)
		out = self.res3(out) + out
		out = self.drop3(out)

		out = self.classifier(out)
		return out

	def predict(self, x):
		x = self.forward(x)
		return x.argmax(1)


model = ResNet(1, 7)
model.load_state_dict(torch.load('ResNet_dict.pth', map_location=torch.device('cpu')))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
count = 0
hist = []

while (True):
	ret, frame = camera.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	faces = face_cascade.detectMultiScale(frame, 1.4, 5)
	for (x, y, w, h) in faces:
		img = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 3)
		f = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
		x = np.array(f).reshape(1,48,48)
		emotion = make_emotion(model,x)
		#cv2.imwrite('face/%s.jpeg' % str(count),f[:,:,1])
		count += 1
		hist.append(emotion)
		cv2.putText(frame,emotion,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,cv2.LINE_AA, False)
	cv2.imshow('camera', frame)
	if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
		with open('outfile', 'wb') as fp:
			pickle.dump(hist, fp)
		break
camera.release()
cv2.destroyAllWindows()

