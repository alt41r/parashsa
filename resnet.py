import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import cv2
for i in range(1):
    img = cv2.imread(f'face/2.jpeg')#you can pass
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #   gray = np.expand_dims(gray, axis=-1)
    Image.fromarray((gray)).show(())

    x=np.array(gray)
    x = x.reshape(1,48,48)
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


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
    model = ResNet(1,7)
    model.load_state_dict(torch.load('ResNet_dict.pth',map_location=torch.device('cpu')))
    def make_emotion(model,x):
        emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        x= torch.tensor(x).unsqueeze(0).float()
        return emotions[model.predict(x).item()]

    print()

