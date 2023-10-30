import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

class ResNet50_v0(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard ResNet50
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, n_class):
        super(ResNet50_v0, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, n_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet50(x)
        return x

class resnet50_mcs(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard ResNet50
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, n_class):
        super(resnet50_mcs, self).__init__()

        A_model = ResNet50_v0(n_class=n_class)
        self.featureA = A_model
        self.classA = nn.Sequential(*list(A_model.resnet50.children())[:-1])

        B_model = ResNet50_v0(n_class=n_class)
        self.featureB = B_model
        self.classB = nn.Sequential(*list(B_model.resnet50.children())[:-1])

        C_model = ResNet50_v0(n_class=n_class)
        self.featureC = C_model
        self.classC = nn.Sequential(*list(C_model.resnet50.children())[:-1])

        num_ftrs = A_model.resnet50.layer4[2].bn3.num_features

        self.combine1 = nn.Sequential(
            nn.Linear(n_class * 4, n_class),
            nn.Sigmoid()
        )

        self.combine2 = nn.Sequential(
            nn.Linear(num_ftrs * 3, n_class),
            nn.Sigmoid()
        )

    def forward(self, x, y, z):
        x1 = self.featureA(x)
        y1 = self.featureB(y)
        z1 = self.featureC(z)
        x2 = self.classA(x)
        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
        y2 = self.classB(y)
        y2 = F.relu(y2, inplace=True)
        y2 = F.adaptive_avg_pool2d(y2, (1, 1)).view(y2.size(0), -1)
        z2 = self.classC(z)
        z2 = F.relu(z2, inplace=True)
        z2 = F.adaptive_avg_pool2d(z2, (1, 1)).view(z2.size(0), -1)

        combine = torch.cat((x2.view(x2.size(0), -1),
                             y2.view(y2.size(0), -1),
                             z2.view(z2.size(0), -1)), 1)
        combine = self.combine2(combine)

        combine3 = torch.cat((x1.view(x1.size(0), -1),
                              y1.view(y1.size(0), -1),
                              z1.view(z1.size(0), -1),
                              combine.view(combine.size(0), -1)), 1)

        combine3 = self.combine1(combine3)

        return x1, y1, z1, combine, combine3

