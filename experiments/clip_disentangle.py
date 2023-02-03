import torch
import clip
from models.base_model import CLIPDomainDisentangleModel
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

class CLIPDisentangleExperiment:
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if  not torch.cuda.is_available() else 'cuda:0')

        # Setup model
        self.model = CLIPDomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Setup optimization procedure and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.reconstructor_loss = torch.nn.MSELoss()
        self.clip_loss = torch.nn.MSELoss()
        self.class_loss_ent = EntropyLoss()
        self.domain_loss_ent = EntropyLoss() 

        self.max_epoch = opt["max_iterations"]
        self.epoch = 0

    def train_iteration(self, data):
        p = self.epoch / self.max_epoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.epoch += 1

        alpha = torch.Tensor([alpha]).to(self.device)
        src_img, src_y, src_d, trg_img, _, trg_d = data
 
        src_img = src_img.to(self.device)
        src_y = src_y.to(self.device)
        src_d = clip.tokenize(src_d, truncate = True).to(self.device)
        src_d = self.clip_model.encode_text(src_d)

        trg_img = trg_img.to(self.device)
        trg_d = clip.tokenize(trg_d, truncate = True).to(self.device)
        trg_d = self.clip_model.encode_text(trg_d)
        
        self.optimizer.zero_grad()

        # Processing a Source Domain Image
        # src_class_out, src_domain_out, src_features_out, src_reconstructor_out = self.model(src_img)
        src_class_output, src_domain_output, src_features, src_reconstructed_features, src_class_output_ds, src_domain_output_cs, src_clip, src_f_ds = self.model(src_img, src_d, alpha, True)
        _, trg_domain_output, trg_features, trg_reconstructed_features, trg_class_output_ds, trg_domain_output_cs, trg_clip, trg_f_ds = self.model(trg_img, trg_d, alpha, True)

        # source class loss
        src_loss_class = self.class_loss(src_class_output, src_y)
        #print(src_loss_class)
        # source domain loss
        # create the expected domain output for source
        # src_domain = 0, so creata a tensor of n=batch_size elements
        src_domain_label = torch.zeros(src_img.shape[0]).long().to(self.device)

        # target domain loss
        trg_domain_label = torch.ones(trg_img.shape[0]).long().to(self.device)

        tot_loss_domain = self.domain_loss(cat((src_domain_output, trg_domain_output), dim=0), cat((src_domain_label, trg_domain_label), dim=0))
        #print(src_loss_domain)
        #print("OK src loss domain")

        # source reconstructor loss
        src_loss_rec = self.reconstructor_loss(src_reconstructed_features, src_features)
        # target reconstructor loss
        trg_loss_rec = self.reconstructor_loss(trg_reconstructed_features, trg_features)

        tot_loss_rec = (src_loss_rec + trg_loss_rec) / 2
        #print(src_loss_rec)
        # entropy loss of class output w.r.t. domain specific features
        src_loss_class_ent = self.class_loss_ent(src_class_output_ds)
        #print(src_loss_class_ent)
        #print("src ent loss cla", src_loss_class_ent)

        #CLIP loss
        src_clip_loss = self.clip_loss(src_clip, src_f_ds)
        trg_clip_loss = self.clip_loss(trg_clip, trg_f_ds)

        tot_clip_loss = (trg_clip_loss + src_clip_loss) / 2

        # entropy loss of domain output w.r.t. class specific features
        #src_loss_domain_ent = self.domain_loss_ent(src_domain_output_cs)
        # compute the domain entropy loss w.r.t. category specific features
        #trg_loss_domain_ent = self.domain_loss_ent(trg_domain_output_cs)

        tot_loss_domain_ent = self.domain_loss_ent(cat((src_domain_output_cs, trg_domain_output_cs), dim=0))

        #print(src_loss_domain_ent)
        #print("src ent loss dom", src_loss_class_ent)

        #tot_src_loss = src_loss_class + src_loss_domain + src_loss_rec + src_loss_domain_ent + src_loss_class_ent

        #print("OK tot loss source")
        # Processing a Target Domain Image
        # exlucding the class loss


        # n.b.: we do not compute the class loss for the target domains images


        
        tot_loss = (0.4 * src_loss_class) + (0.08 * tot_loss_domain) + (0.02 * tot_loss_rec) + (0.4 * src_loss_class_ent) + (0.08 * tot_loss_domain_ent) + (0.02 * tot_clip_loss)

        

        #print("trg loss dom ent", trg_loss_domain)
        # compute the class entropy loss w.r.t. domain specific features
        # we can still compute it since this loss doesn't require the ground truth to be computed ?? ASK
        #trg_loss_class_ent = self.class_loss_ent(trg_class_output_ds)
        #print("trg loss class ent", trg_loss_class_ent)

        #tot_trg_loss = trg_loss_domain + trg_loss_rec + trg_loss_domain_ent #+ trg_loss_class_ent
        # compute the final loss and backward it
        #tot_loss = tot_src_loss + tot_trg_loss
        tot_loss.backward()
        self.optimizer.step()
        return tot_loss.item()

    # move the checkpoint methods in an abstract class
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

        alpha = torch.Tensor(1).to(self.device)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x,alpha, alpha, False)[0]
                loss += self.class_loss(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss