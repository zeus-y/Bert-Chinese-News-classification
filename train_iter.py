from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
'''
本函数主要是用来写一部分公用的方法
'''


def train(config, model, data, name, test_dataloader, eva_dataloader, logger):
    writer = SummaryWriter(log_dir=config.log_path + '/' + name + '-' +
                           time.strftime('%m-%d_%H.%M', time.localtime()))
    logger.add_log(f'SummaryWriter准备完毕，存储地址为 "./log/SummaryWriter', 'debug')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    if config.cuda_is_aviable:
        model = model.cuda(device=config.cuda_device)
    for epoch in range(config.epoch):
        for i, batch in enumerate(data):
            x, y = batch
            model.zero_grad()
            predidct = model(x)
            loss = loss_function(predidct, y)
            loss.backward()
            optimizer.step()
            eva_accurate, eva_loss = evaluate(
                model=model, data_iter=eva_dataloader)
            print(
                f'epoch: {epoch}, batch: {i}, loss: {loss.item()}, evaluate-accurate: {eva_accurate}, evaluate-loss: {eva_loss}')
            writer.add_scalar('loss-epoch/train', loss.item(), epoch)
            writer.add_scalar('loss-i/train', loss.item(), epoch * 18+i)
        if epoch % 1 == 0:
            _, accurate, f1 = test(model=model, data=test_dataloader)
            print(f'epoch: {epoch}, accurate: {accurate}, f1-score: {f1}')
            writer.add_scalar('accurate/train', accurate, epoch)
            writer.add_scalar('f1-socre', f1, epoch)
            writer.add_scalar('accurate/evaluate', eva_accurate, epoch)
            writer.add_scalar('loss/evaluate', eva_loss, epoch)
            logger.add_log(
                f'epoch: {epoch}, accurate: {accurate}, f1-score: {f1}', 'debug')

    # 训练完保存模型并给出测试结果。
    net = {'net': model.state_dict()}
    torch.save(net, config.model_save_path)
    logger.add_log(f'模型保存结束. 模型地址：{config.model_save_path} ', 'debug')

    report, accurate, f1 = test(model=model, data=test_dataloader)
    logger.add_log(
        f'训练结果: accurate: {accurate}, f1-score: {f1}, report: \n {report}', 'debug')
    print(report)


def test(model, data):
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, batch in enumerate(data):
            x, y = batch
            predict = model(x)
            predict = torch.argmax(predict, dim=1)
            predict = predict.cpu().numpy()
            y = y.cpu().numpy()
            predict_all = np.append(predict_all, predict)
            labels_all = np.append(labels_all, y)
    report = classification_report(predict_all, labels_all)
    # print(report)
    accurate = accuracy_score(predict_all, labels_all)
    f1 = f1_score(predict_all, labels_all, average='micro')
    return report, accurate, f1


def evaluate(model, data_iter):
    # model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter)
