import torch
from models.base_model import DomainDisentangleModelDG
from torch import nn
from torch.nn import functional as F
from torch import cat
import numpy as np
import clip

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = 1.0 * b.sum()
        return b / x.size(0)

class CLIPDomainDisentangleDGExperiment:

    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if  not torch.cuda.is_available() else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModelDG()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # clip model
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu')
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Setup optimization procedure and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.reconstructor_loss = torch.nn.MSELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.class_loss_ent = EntropyLoss()
        self.domain_loss_ent = EntropyLoss() 

    def train_iteration(self, data):
        src_img, src_y, src_desc, src_d = data

        src_img = src_img.to(self.device)
        src_y = src_y.to(self.device)
        src_d = src_d.to(self.device)

        self.optimizer.zero_grad()

        # Processing a Source Domain Image
        src_class_output, src_domain_output, src_features, src_reconstructed_features, src_class_output_ds, src_domain_output_cs, f_ds = self.model(src_img)

        src_loss_class = self.class_loss(src_class_output, src_y)

        src_loss_domain = self.domain_loss(src_domain_output, src_d)

        # source reconstructor loss
        src_loss_rec = self.reconstructor_loss(src_reconstructed_features, src_features)

        # entropy loss of class output w.r.t. domain specific features
        src_loss_class_ent = self.class_loss_ent(src_class_output_ds)

        src_loss_domain_ent = self.domain_loss_ent(src_domain_output_cs)

        desc_token = clip.tokenize(src_desc, truncate=True).to(self.device)
        desc_feat = self.clip_model.encode_text(desc_token).to(self.device)
        clip_loss = self.mse_loss(f_ds, desc_feat)

        tot_loss = (0.4 * src_loss_class) + (0.09 * src_loss_domain) + (0.02 * src_loss_rec) + (0.4 * src_loss_class_ent) + (0.09 * src_loss_domain_ent) + (0.02 * clip_loss)

        tot_loss.backward()
        self.optimizer.step()
        return tot_loss.item()

    # move the checkpoint methods in an abstract class
    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)[0]
                loss += self.class_loss(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss