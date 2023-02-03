import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.autograd import Function

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        y = x.squeeze()
        if (len(y.size()) < 2):
          y = y.unsqueeze(0)
        return y

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        # domain encoder
        # output features used to discriminate the domain
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # category encoder
        # return features used to discriminate the input category
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # domain classifier
        self.domain_classifier = nn.Linear(512, 2)

        # category classifier
        self.category_classifier = nn.Linear(512, 7)

        # reconstructor
        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def set_requires_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
        return

    def forward(self, x):
        features = self.feature_extractor(x)
        f_cs = self.category_encoder(features)
        f_ds = self.domain_encoder(features)
        reconstructed_features = self.reconstructor(torch.cat((f_cs, f_ds), 1))
        class_output = self.category_classifier(f_cs)
        domain_output = self.domain_classifier(f_ds)
        
        self.set_requires_grad(self.category_classifier, False)
        class_output_ds = self.category_classifier(f_ds)
        self.set_requires_grad(self.category_classifier, True)

        self.set_requires_grad(self.domain_classifier, False)
        domain_output_cs = self.domain_classifier(f_cs)
        self.set_requires_grad(self.domain_classifier, True)

        return class_output, domain_output, features, reconstructed_features, class_output_ds, domain_output_cs

class CLIPDomainDisentangleModel(nn.Module):
    def __init__(self):
        super(CLIPDomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        # domain encoder
        # output features used to discriminate the domain
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU()
        )

        # category encoder
        # return features used to discriminate the input category
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # domain classifier
        self.domain_classifier = nn.Linear(512, 2)

        # category classifier
        self.category_classifier = nn.Linear(512, 7)

        # reconstructor
        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x, descr, alpha, train):

        features = self.feature_extractor(x)
        f_cs = self.category_encoder(features)
        f_ds = self.domain_encoder(features)
        reconstructed_features = self.reconstructor(torch.cat((f_cs, f_ds), 1))
        class_output = self.category_classifier(f_cs)
        domain_output = self.domain_classifier(f_ds)
        class_output_ds = self.category_classifier(f_ds)
        domain_output_cs = self.domain_classifier(f_cs)

        return class_output, domain_output, features, reconstructed_features, class_output_ds, domain_output_cs
        