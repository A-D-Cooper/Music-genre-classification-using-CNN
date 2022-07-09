from torch import nn


class CNN_classifier(nn.Module):

    def __init__(self, c=6, n=800, n2=600):
        super(CNN_classifier, self).__init__()

        self.conv1 = nn.Conv2d(3, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(c * 2)

        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(c * 4)

        self.conv4 = nn.Conv2d(c * 4, c * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.f = nn.Flatten()
        self.r = nn.ReLU()
        self.s = nn.Softmax(dim=1)

        self.l1 = nn.Linear(c * 8 * 14 * 14, n)
        self.l2 = nn.Linear(n, n2)
        self.l3 = nn.Linear(n2, 9)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.r(c1)
        p1 = self.pool1(c1)
        bn1 = self.bn1(p1)

        c2 = self.conv2(bn1)
        c2 = self.r(c2)
        p2 = self.pool2(c2)
        bn2 = self.bn2(p2)

        c3 = self.conv3(bn2)
        c3 = self.r(c3)
        p3 = self.pool3(c3)
        bn3 = self.bn3(p3)

        c4 = self.conv4(bn3)
        c4 = self.r(c4)
        p4 = self.pool4(c4)

        f = self.f(p4)

        l1 = self.l1(f)
        r1 = self.r(l1)

        l2 = self.l2(r1)
        r2 = self.r(l2)

        l3 = self.l3(r2)
        s = self.s(l3)

        return s