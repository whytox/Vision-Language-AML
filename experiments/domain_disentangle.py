import torch
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.entropy_criterion = torch.nn.CrossEntropyLoss()
        self.reconstructor_criterion = torch.nn.MSELoss()

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

    def train_iteration(self, data):
        x,y = data
        x = x.to(self.device)
        y = y.to(self.device)

        out_0 = self.model(x,0)

        if y != -1: #Processing a Source Domain Image
            d = 0
            out_1 = self.model(x,1)
            loss_1 = self.entropy_criterion(out_1, y)
            out_2 = self.model(x,2)
            loss_2 = - self.entropy_criterion(out_2, d)
            out_3 = self.model(x,3)
            loss_3 = self.entropy_criterion(out_3, d)
            out_4 = self.model(x,4)
            loss_4 = - self.entropy_criterion(out_4, y)
            out_5 = self.model(x,5)
            loss_5 = self.reconstructor_criterion(out_5, out_0)

            self.optimizer.zero_grad()
            
            loss_1.backward(retain_graph = True)
            loss_2.backward(retain_graph = True)
            loss_3.backward(retain_graph = True)
            loss_4.backward(retain_graph = True)
            loss_5.backward(retain_graph = True)

            self.optimizer.step()

            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

            return loss.item()

        else : #Processing a Target Domain Image
            d = 1
            out_2 = self.model(x,2)
            loss_2 = - self.entropy_criterion(out_2, d)
            out_3 = self.model(x,3)
            loss_3 = self.entropy_criterion(out_3, d)
            out_5 = self.model(x,5)
            loss_5 = self.reconstructor_criterion(out_5, out_0)

            self.optimizer.zero_grad()

            loss_2.backward(retain_graph = True)
            loss_3.backward(retain_graph = True)
            loss_5.backward(retain_graph = True)

            loss = loss_2 + loss_3 + loss_5

            return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if y != -1: #Processing a Source Domain Image
                    out = self.model(x,1)
                    loss += self.entropy_criterion(out, y)
                    pred = torch.argmax(out, dim=-1)

                    accuracy += (pred == y).sum().item()
                    count += x.size(0)

                else : #Proccessing a Target Domain Image
                    out = self.model(x,3)
                    loss += self.entropy_criterion(out, 1)
                    pred = torch.argmax(out, dim=-1)

                    accuracy += (pred == y).sum().item()
                    count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss