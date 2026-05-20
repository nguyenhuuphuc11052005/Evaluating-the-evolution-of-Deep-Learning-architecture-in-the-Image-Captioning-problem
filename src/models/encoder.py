import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    



class ResNet50SpatialEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ResNet50SpatialEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Bỏ đi 2 lớp cuối cùng: AdaptiveAvgPool2d và Linear
        # Đầu ra lúc này sẽ là các feature map không gian (spatial features)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Dùng Conv2d 1x1 để giảm chiều sâu từ 2048 xuống embed_size cho nhẹ
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1)
        
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        # images: (batch_size, 3, 224, 224)
        features = self.resnet(images)  # Output: (batch_size, 2048, 7, 7)
        features = self.conv(features)  # Output: (batch_size, embed_size, 7, 7)
        
        # Làm phẳng lưới 7x7 thành 49 vùng (sequence length = 49)
        features = features.flatten(2)  # Output: (batch_size, embed_size, 49)
        features = features.permute(0, 2, 1) # Output: (batch_size, 49, embed_size)
        
        return features # Đây chính là chuỗi 49 "từ vựng thị giác" để đưa vào Transformer