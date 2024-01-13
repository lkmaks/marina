from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
import threading
from debug_utils import *
from param_utils import *
from WorkerThread import WorkerThread


def CalculateGrad(loader, model, device, coeff=1):
    for inputs, outputs in loader:
        inputs, outputs = inputs.to(device), outputs.to(device)  # move to device
        logits = model(inputs)  # forward-pass: Make a forward pass through the network
        loss = coeff * F.cross_entropy(logits, outputs, reduction='sum')  # compute objective
        loss.backward()  # compute the gradient (backward-pass)


class WorkerThreadMarina():
    def __init__(self, wcfg, ncfg, stats, print_lock=None, model=None):
        super(WorkerThreadMarina, self).__init__(wcfg, ncfg, stats, print_lock, model)

    def run(self):
        wcfg = self.wcfg
        ncfg = self.ncfg

        # wcfg - configuration specific for worker
        # ncfg - general configuration with task description

        dbgprint(self.print_lock, wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ",
                 len(wcfg.train_set))

        model_x = self.model
        model_w = copy_module(model_x)

        one_div_trainset_len = torch.Tensor([1.0 / len(wcfg.train_set)]).to(wcfg.device)
        one_div_batch_prime_len = torch.Tensor([1.0 / (ncfg.batch_size_for_worker)]).to(wcfg.device)

        delta_flatten = torch.zeros(ncfg.D).to(wcfg.device)
        # =========================================================================================
        iteration = 0
        # =========================================================================================

        while True:
            wcfg.input_cmd_ready.acquire()
            if wcfg.cmd == "exit":
                wcfg.output_of_cmd = ""
                wcfg.cmd_output_ready.release()
                break
            elif wcfg.cmd == "set_x":
                model_x = copy_module(wcfg.input_for_cmd)
            elif wcfg.cmd == "set_w":
                model_w = copy_module(wcfg.input_for_cmd)
            elif wcfg.cmd == "get_grad_diff":
                CalculateGrad(wcfg.train_loader, model_x, wcfg.device, one_div_trainset_len)
                CalculateGrad(wcfg.train_loader, model_w, wcfg.device, one_div_trainset_len)

        # Signal that worker has finished initialization via decreasing semaphore
        # completed_workers_semaphore.acquire()
        dbgprint(self.print_lock, wcfg, f"END")
