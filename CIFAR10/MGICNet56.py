from __future__ import division
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from mgic.mgic import Net56


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fun = nn.CrossEntropyLoss()


def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49140054, 0.48215815, 0.44652817],
                             std=[0.24703224, 0.24348514, 0.26158786]),
    ])

    trainset = datasets.CIFAR10(root='../../data', train=True,
                                download=True, transform=transform)
    testset = datasets.CIFAR10(root='../../data', train=False,
                               download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=8, pin_memory=True)
    return train_loader, test_loader


def train(epochs, train_loader, test_loader):
    loss_val = []
    loss_train = []
    avg_loss_train = []
    accur = []
    accur_train = []
    model = Net56(10, 3)
    # model = ResNet18()
    # model = models.vgg16()
    opt = optim.Adam(model.parameters(), lr=0.001,
                     betas=(0.9, 0.999), eps=1e-08,
                     weight_decay=0, amsgrad=True
                     )
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            output = model(data)
            loss = loss_fun(output, target)
            total_train_loss += loss * len(data)
            loss_train.append(loss)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accur_train.append(correct / len(data))

        model.eval()
        test_loss = 0.
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.float(), target.to(dtype=torch.long)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fun(output, target).item() * len(data)  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        loss_val.append(test_loss)
        avg_loss_train.append(total_train_loss / len(train_loader.dataset))
        test_acc = correct / len(test_loader.dataset)
        accur.append(test_acc)
        sch.step(test_loss)
        print('\nTrain set: Average loss: {:.8f}\n'.format(total_train_loss / len(train_loader.dataset)))

        print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return loss_val, loss_train, avg_loss_train, accur, accur_train


def main():
    bs = 64
    t_l, tst_l = get_data(bs)
    eps = 100

    l_v, l_t, al_t, acc, acct = train(eps, t_l, tst_l)

    np.save("./validation_losses.txt", np.asarray(l_v))
    np.save("./training_losses.txt", np.asarray(l_t))
    np.save("./average_training_losses.txt", np.asarray(al_t))
    np.save("./accuracy.txt", np.asarray(acc))
    np.save("./accuracy_training.txt", np.asarray(acct))


if __name__ == '__main__':
    main()
