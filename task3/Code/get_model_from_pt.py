import torchvision.models as models
import torch
import torch.nn as nn

vgg16 = models.vgg16(pretrained=True)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool  # output 512x7x7 = 25'088 output nodes
        self.classifier = vgg16.classifier
        self.classifier[0] = nn.Linear(25088 * 3, 4096, bias=True)
        self.classifier[6] = nn.Linear(4096, 1, bias=True)
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    # x represents our data
    def forward(self, x):

        with torch.no_grad():
            x = torch.reshape(x, (x.shape[0] * 3,) + x[0][0].shape)

            x = self.features(x)
            x = self.avgpool(x)

            x = torch.reshape(x, (x.shape[0] // 3, 3) + x[0].shape)
            x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x

model = Net()
checkpoint = torch.load('model_epoch_40.pt',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
torch.save(model,'vgg_epoch_40.pth')