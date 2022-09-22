import torch


class Loop:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self, epoch):
        with torch.no_grad():
            self.model.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # sum up batch loss
                test_loss += self.loss_fn(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)
            print('Test Epoch:{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(epoch, test_loss, correct, len(self.test_loader.dataset),
                          100. * correct / len(self.test_loader.dataset)))
