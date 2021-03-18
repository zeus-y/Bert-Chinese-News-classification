import logging
import logging.handlers
import datetime

'''
    function：记录日志
    author：wcy
    date：2020.1.9
    apply: Logging.__init__()   Logging.add_log()
'''


class Logging():
    @classmethod
    def __init__(cls, name, save_path='', logerlevel='DEBUG'):
        '''
            path: 存储文件的路径
            logerlevel：表示记录的日志的等级, 默认值为{DEBUG, INFO ,WARNING , ERROR ,CRITICAL }
        '''
        cls.logger = logging.getLogger(name)
        if hasattr(logging, logerlevel):
            print(getattr(logging, logerlevel))
            cls.logger.setLevel(getattr(logging, logerlevel))
            print(f'**************设置日志等级为:{logerlevel}*****************')
        else:
            cls.logger.setLevel(logging.DEBUG)
            print(f'***************设置日志等级为:DEBUG******************')
        all_path = save_path + name + '-all.log'
        rf_handler = logging.handlers.TimedRotatingFileHandler(all_path, when='midnight', interval=1,
                                                               backupCount=7, atTime=datetime.time(0, 0, 0, 0))
        rf_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"))

        error_path = save_path + name + '-error.log'
        f_handler = logging.FileHandler(error_path)
        f_handler.setLevel(logging.ERROR)
        f_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

        cls.logger.addHandler(rf_handler)
        cls.logger.addHandler(f_handler)

        print('*****************开始记录日志********************')

    @classmethod
    def add_log(cls, info, level='debug'):
        '''
            info: 要记录的日志信息
            level: 要记录的日志的等级    默认值：{info, debug, warning, error, critical}
            function：本函数主要是利用反射方法记录日志
        '''
        if hasattr(cls.logger, level):
            log = getattr(cls.logger, level)
            log(info)
        else:
            cls.debug(info)
            cls.error(f'选择了错误的日志等级{level}')


if __name__ == "__main__":
    Logging.__init__()
    Logging.add_log('我是测试数据', 'debug')
