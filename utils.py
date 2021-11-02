import os
import psutil
import sys
import json
import logging
import torch
from tensorboardX import SummaryWriter
from argparse import Namespace

def get_logger(fname=None, stdout=True):
    '''
    fname: file location to store the log file
    '''
    handlers = []
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)
    if fname:
        file_handler = logging.FileHandler(filename=fname)
        handlers.append(file_handler)

    str_fmt = '[%(asctime)s.%(msecs)03d] %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=str_fmt,
        datefmt=date_fmt,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger

def setup_experiment_log(args, savedir='./results/', exp_name='test', save=False):
    '''
    savedir: str location to save contents in
    save: boolean
    Returns: tuple of str (log file) and SummaryWriter
    '''
    if not save:
        return None, None

    if os.path.exists(savedir):
        exp_dir = os.path.join(savedir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        sumdir = os.path.join(exp_dir, 'summary')
        swr = SummaryWriter(sumdir)
        json.dump(args.__dict__, open(os.path.join(exp_dir, 'args.json'), 'w'))
        logfile = os.path.join(exp_dir, 'output.log')
        cnt = 1
        while os.path.exists(logfile):
            logfile = os.path.join(exp_dir, f'output{cnt}.log')
            cnt += 1

    else:
        # make the save dir, retry
        os.makedirs(savedir)
        return setup_experiment_log(savedir, exp_name, save)

    return logfile, swr

def check_memory(verbose=True):
    '''
    verbose: bool, flag to pretty print consumed memory
    Returns: memory usage in MB
    '''
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    if verbose:
        print("Consumed {:.2f}mb memory".format(mem))
    return mem

def save_checkpoint(epoch, model, optimizer, fname):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, fname)

def load_checkpoint(model, optimizer, log, fname):
    start_epoch = 0
    if os.path.isfile(fname):
        log.info("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(fname, checkpoint['epoch']))
        success = True
    else:
        log.info("=> no checkpoint found at '{}'".format(fname))
        success = False

    return model, optimizer, start_epoch, success
