import os
import logging
import torch
import numpy as np
from termcolor import colored
from .rank_filter import rank_filter
from .path import mkdir


class Logger:
    def __init__(self, local_rank, save_dir='./', use_tensorboard=True):
        mkdir(local_rank, save_dir)
        self.rank = local_rank
        # 设置命令行窗口输出日志的字体颜色
        fmt = colored('[%(name)s]', 'magenta', attrs=['bold']) + colored('[%(asctime)s]', 'blue') + \
              colored('%(levelname)s:', 'green') + colored('%(message)s', 'white')
        logging.basicConfig(level=logging.INFO,  # 设置保存到logs.txt里的文本是INFO级别的
                            filename=os.path.join(save_dir, 'logs.txt'),
                            filemode='w')
        self.log_dir = os.path.join(save_dir, 'logs')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
            if self.rank < 1:
                logging.info('Using Tensorboard, logs will be saved in {}'.format(self.log_dir))
                self.writer = SummaryWriter(log_dir=self.log_dir)

    # 训练过程中qfl、dfl、bbox损失的显示靠的是调用下面的log方法
    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    # 从它的usages可以看出，tensorboard里的训练记录是通过调用scalar_summary实现
    def scalar_summary(self, tag, phase, value, step):
        if self.rank < 1:
            self.writer.add_scalars(tag, {phase: value}, step)


class MovingAverage(object):
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val):
        self.reset()
        self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


# Source from : https://github.com/ultralytics/yolov5/blob/master/utils/general.py
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']