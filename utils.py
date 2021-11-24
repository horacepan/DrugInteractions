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
    Returns: logger object

    Use the logger object anywhere where you might use a print statement. The logger
    object will print log messages to stdout in addition to writing it to a log file.
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
    exp_name: Name of experiment
    save: boolean
    Returns: tuple of str (log file) and SummaryWriter
        SummaryWriter will write to the location specified by: {savedir}/{exp_name}/summary
        The returned logfile string will be: {savedir}/{exp_name}/output.log
        If the existing logfile exists (suppose you want to rerun an experiment with
        the model reloaded from a save checkpoint to continue training), the output log
        file will instead be output1.log/output2.log/etc.
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
        return setup_experiment_log(args, savedir, exp_name, save)

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
    '''
    epoch: int, epoch number
    model: nn.Module, the model to save
    optimizer: torch.optim optimizer object
    fname: string, location to save the checkpoint file to

    Saves a state dict of the model and optimizer so that it can be reloaded.
    '''
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, fname)

def load_checkpoint(model, optimizer, log, fname):
    '''
    model: nn.Module, the model to reload
    optimizer: torch.optim optimizer, the optizer to reload
    log: logger object
    fname: string, file name to load checkpoint from

    Returns: tuple of the reloaded model, optimizer, epoch of the checkpoint, and a boolean
        indicating whether or not the checkpoint was succesffully loaded.
    Loads the state dict for the given model, optimizer
    '''
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
