import torch
import torch.nn as nn
from torchvision.models import resnet18

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

class GradientReversal(Function):
    """The GradientReversal layer used to reverse the gradient 
    in adversarial training step."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None

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

            nn.Linear(512, 256), # changed 512 to 256
            nn.BatchNorm1d(256), # changed 512 to 256
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

            nn.Linear(512, 256), # changed 512 to 256
            nn.BatchNorm1d(256), # changed 512 to 256
            nn.ReLU()
        )

        # domain classifier
        self.domain_classifier = nn.Sequential(nn.Linear(256, 64), nn.LeakyReLU(),
                                               nn.Linear(64, 2), nn.Softmax())

        # category classifier
        self.category_classifier = nn.Sequential(nn.Linear(256, 7), nn.BatchNorm1d(7), nn.Softmax())

        # reconstructor
        self.reconstructor = nn.Sequential(
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

    def forward(self, x, alpha):

        features = self.feature_extractor(x)
        #print("OK 1")
        f_cs = self.category_encoder(features)
        #print("OK 2")
        f_ds = self.domain_encoder(features)
        #print("OK 3")
        reconstructed_features = self.reconstructor(torch.cat((f_cs, f_ds), 1))
        #print("OK 4")
        class_output = self.category_classifier(f_cs)
        #print("OK 5")
        domain_output = self.domain_classifier(f_ds)
        #print("OK 6")
        # in order to compute the gradient reverse 
        # we let the features pass from the GradientReversal layer first
        
        reverse_grad_f_cs = GradientReversal.apply(f_cs, alpha)
        #print("OK 7")
        #print("OK 8")
        reverse_grad_f_ds = GradientReversal.apply(f_ds, alpha)
        class_output_ds = self.category_classifier(reverse_grad_f_ds)

        #print("OK 9")
        domain_output_cs = self.domain_classifier(reverse_grad_f_cs)
        #print("OK 10")

        return class_output, domain_output, features, reconstructed_features, class_output_ds, domain_output_cs