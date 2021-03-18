import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig


class Config():
    def __init__(self) -> None:
        self.name = 'bert'
        self.train_path = 'data/train.txt'
        self.dev_path = 'data/dev.txt'
        self.test_path = 'data/test.txt'
        self.log_path = './log/SummaryWriter'
        self.model_save_path = f'./save/{self.name}.pkl'
        self.device = 2
        self.shuffle = True                                                # 加载数据时是否随机加载
        self.cuda_is_aviable = True                                       # 是否可以GPU加速
        self.cuda_device = 2                                               # 指定训练的GPU
        self.learning_rate = 1e-5                                          # 学习率的大小
        self.epoch = 10
        self.pad_size = 32
        self.batch_size = 256
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
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
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
            Bert最终输出的结果维度为：sequence_output, pooled_output, (hidden_states), (attentions)
            以输入序列为19为例：
            sequence_output：torch.Size([1, 19, 768])
            输出序列
            pooled_output：torch.Size([1, 768])
            对输出序列进行pool操作的结果
            (hidden_states)：tuple, 13 * torch.Size([1, 19, 768])
            隐藏层状态（包括Embedding层），取决于 model_config 中的 output_hidden_states
            (attentions)：tuple, 12 * torch.Size([1, 12, 19, 19])
            注意力层，取决于 model_config 中的 output_attentions
        '''
        context = x[0]  # 输入的句子
        # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[2]
        pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled[1])
        out = self.softmax(out)
        return out
