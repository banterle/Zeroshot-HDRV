import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def conv1to3(input):
    sz = input.shape
    if sz[1] == 1:
        tmp = torch.zeros((sz[0], 3, sz[2], sz[3]))
        tmp[:,0,:,:] = input[:,0,:,:]
        tmp[:,1,:,:] = input[:,0,:,:]
        tmp[:,2,:,:] = input[:,0,:,:]
        
        if torch.cuda.is_available():
            input = tmp.cuda()
        else:
            input = tmp
    
    return input

class VGG16Loss(nn.Module):
    def __init__(self, layers_weights=None):
        super(VGG16Loss, self).__init__()
        self.vgg = models.vgg16(weights='DEFAULT').features.eval()

        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # Define layers to extract features from (names based on VGG16 structure)
        self.layer_ids = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        self.layers_weights = layers_weights or [1.0, 1.0, 1.0, 1.0]
        self.criterion = nn.L1Loss()  # You can switch to MSELoss if preferred

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # VGG16 expects normalized inputs
            std=[0.229, 0.224, 0.225]
        )
        
    def forward(self, input, target):
        input = conv1to3(input)
        target = conv1to3(target)

        # Normalize input and target images
        input = self._normalize_batch(input)
        target = self._normalize_batch(target)

        loss = 0.0
        x = input
        y = target

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)

            if i in self.layer_ids:
                idx = self.layer_ids.index(i)
                loss += self.layers_weights[idx] * self.criterion(x, y)

        return loss

    def _normalize_batch(self, batch):
        # Assume input is in range [0, 1], shape: (B, 3, H, W)
        normalized = torch.clone(batch)
        for i in range(3):
            normalized[:, i] = (normalized[:, i] - self.normalize.mean[i]) / self.normalize.std[i]
        return normalized

#
#
#
if __name__ == "__main__":
    imgA = Image.open("testA.png").convert("RGB")
    imgB = Image.open("testB.png").convert("RGB")

    transform = transforms.ToTensor()
    imgA_t = transform(imgA)
    imgB_t = transform(imgB)
    
    loss = VGG16Loss()
    print(loss(imgA_t, imgB_t))
    

