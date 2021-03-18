from time import time
from utils import DatasetIterater, build_dataset, build_iterator, get_time_dif
from train_iter import train
from LoggerClass import Logging
from importlib import import_module
import argparse


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model
    x = import_module(f'model.{model_name}')

    Logging.__init__(model_name, save_path='./log/', logerlevel='DEBUG')
    Logging.add_log(f'读取命令行输入的参数: model: {model_name}', 'debug')

    config = x.Config()
    model = x.Model(config)
    start_time = time()

    train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 开始训练：
    # train(config, model, train_iter, dev_iter, test_iter)
    train(config, model, train_iter, model_name, test_iter, dev_iter, Logging)
