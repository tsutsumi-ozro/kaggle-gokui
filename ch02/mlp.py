import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn, optim

def seed_torch(seed=1485):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':
    seed_torch()

    digits = load_digits()
    X = digits.data
    y = digits.target
    print(X.shape, y.shape)
    #(1797, 64) (1797, )

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    model.train()
    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = []

    for ep in range(100):
        optimizer.zero_grad()
        out = model(X)

        loss = lossfun(out, y)
        loss.backward()

        optimizer.step()

        losses.append(loss.item())
    # ＿にはmaxの値, predにmaxのindexが格納される
    # https://pytorch.org/docs/stable/generated/torch.max.html
    _, pred = torch.max(out, 1)
    print((pred==y).sum().item() / len(y))

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('ch02/output_ch02/fig00.png')
