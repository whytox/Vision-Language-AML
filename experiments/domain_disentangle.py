import torch
from models.base_model import DomainDisentangleModel
from torch import nn
from torch.nn import functional as F
from torch import cat
import numpy as np


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = 1.0 * b.sum()
        return b / x.size(0)

class DomainDisentangleExperiment:
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if  not torch.cuda.is_available() else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.reconstructor_loss = torch.nn.MSELoss()
        self.class_loss_ent = EntropyLoss()
        self.domain_loss_ent = EntropyLoss() 

        self.max_epoch = opt["max_iterations"]
        self.epoch = 0

    def train_iteration(self, data):
        x_src, y_src, x_trg, _ = data
        x_src, y_src, x_trg = x_src.to(self.device), y_src.to(self.device), x_trg.to(self.device)

        x = torch.cat((x_src, x_trg), dim=0)
        
        src_dom_label = torch.zeros(x_src.size(0)).long()
        src_dom_label = src_dom_label.to(self.device)
        trg_dom_label = torch.ones(x_trg.size(0)).long()
        trg_dom_label = trg_dom_label.to(self.device)
        dom_label     = torch.cat((src_dom_label, trg_dom_label), 0)
        self.optimizer.zero_grad()

        cat, dom, feat, rec_feat, cat_ds, dom_cs = self.model(x)

        cat_loss     = self.cross_ent_loss(cat[:x_src.size(0)], y_src)
        dom_loss     = self.cross_ent_loss(dom, dom_label)
        cat_ent_loss =   self.entropy_loss(cat_ds)
        dom_ent_loss =  self.entropy_loss(dom_cs) 
        rec_loss     = self.reconstructor_loss(rec_feat, feat) 
        
        tot_loss = 0.4 * cat_loss + 0.09 * dom_loss + 0.01 * rec_loss + 0.4 * cat_ent_loss + 0.090 * dom_ent_loss 
        tot_loss.backward()
        self.optimizer.step()
        
        return tot_loss.item()

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