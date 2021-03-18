import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig


class Config():
    def __init__(self) -> None:
        self.name = 'bertRCNN'
        self.train_path = 'data/train.txt'
        self.dev_path = 'data/dev.txt'
        self.test_path = 'data/test.txt'
        self.log_path = './log/SummaryWriter'
        self.model_save_path = f'./save/{self.name}.pkl'
        self.shuffle = True                                                # 加载数据时是否随机加载
        self.cuda_is_aviable = True                                        # 是否可以GPU加速
        self.cuda_device = 0                                               # 指定训练的GPU
        self.learning_rate = 1e-5                                          # 学习率的大小
        self.epoch = 10
        self.pad_size = 32
        self.batch_size = 256
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.out = 256
        self.hidden_num = 2
        self.dropout = 0.5
        self.num_classes = 10


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_config = BertConfig.from_pretrained(config.bert_path)
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = True
        self.bert = BertModel.from_pretrained(
            config.bert_path, config=self.model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.gru = nn.GRU(input_size=config.hidden_size,
                          hidden_size=config.out, num_layers=config.hidden_num, batch_first=True, dropout=config.dropout, bidirectional=True)
        self.max_pool = nn.MaxPool1d(config.pad_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.out * 2, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        context = x[0]  # 输入的句子
        # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[2]
        pooled = self.bert(context, attention_mask=mask)
        out = pooled[0]
        out, _ = self.gru(out)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out = self.max_pool(out)
        out = out.squeeze()
        out = self.relu(self.fc(out))
        out = self.softmax(out)
        return out
