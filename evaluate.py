import torch

from model import Net
from dataLoader import dataLoader

trainloader, testloader, classes = dataLoader()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().to(device)
net.load_state_dict(torch.load('save.pth'))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

