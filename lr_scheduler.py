import torch
import torch.optim as optim
import matplotlib.pyplot as plt

LR = 0.1  # 设置初始学习率
iteration = 10
max_epoch = 200

# --------- fake data and optimizer  ---------
weights = torch.randn((1), requires_grad=True)
target = torch.zeros((1))

# 构建虚拟优化器，为了lr_scheduler关联优化器
optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

# ------------- 3 Exponential LR -----------
# flag = 0
flag = 1
if flag:
    gamma = 0.95
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)

            loss.backward()
            # 优化器参数更新
            optimizer.step()
            optimizer.zero_grad()
            # 学习率更新
        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Exponential LR Scheduler\ngamma:{}".format(gamma))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.savefig('./ExponentialLR.jpg')
    # plt.show()
