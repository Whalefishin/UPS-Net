import os
import sys
import torch
import pickle
import gzip

class Logger(object):

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
class Plot_logger(object):
    def __init__(self, root_path=None):
        self.root_path = root_path

    def log(self, data_name, data):
        fpath = os.path.join(self.root_path, data_name + '.txt')
        self.file = open(fpath, 'a')
        self.file.write(str(data) + '\n')
    