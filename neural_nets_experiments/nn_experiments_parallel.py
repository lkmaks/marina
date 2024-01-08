#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

# PyTorch modules
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import random

from time import time
from datetime import datetime
import threading


# Utils
import utils
import compressors
from param_utils import *


#====================================================================================
class NNConfiguration: pass
class WorkersConfiguration: pass

class AlgoNames:
    DIANA = 0
    VR_DIANA = 1
    MARINA = 2
    VR_MARINA = 3

#====================================================================================
transfered_bits_by_node = None
fi_grad_calcs_by_node   = None
train_loss              = None
test_loss               = None
train_acc               = None
test_acc                = None
fn_train_loss_grad_norm = None
fn_test_loss_grad_norm  = None


#====================================================================================
# MULTITHREAD DEBUG

print_lock = threading.Lock()

def dbgprint(wcfg, *args):
    printing_dbg = True
    if printing_dbg == True:
        print_lock.acquire()
        print(f"Worker {wcfg.worker_id}/{wcfg.total_workers}:", *args, flush = True)
        print_lock.release()

def rootprint(*args):
    print_lock.acquire()
    print(f"Master: ", *args, flush = True)
    print_lock.release()

# ====================================================================================
# Statistics

def getAccuracy(model, trainset, batch_size, device):
    avg_accuracy = 0

    dataloader = DataLoader(
                trainset,                  # dataset from which to load the data.
                batch_size=batch_size,     # How many samples per batch to load (default: 1).
                shuffle=False,             # Set to True to have the data reshuffled at every epoch (default: False)
                drop_last=False,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                pin_memory=False,          # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                collate_fn=None,           # Merges a list of samples to form a mini-batch of Tensor(s)
            )

    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)           
    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)                             # move to device
        logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
        avg_accuracy += (logits.data.argmax(1) == outputs).sum().item()

    avg_accuracy /= len(trainset)
    model.train(prev_train_mode)

    return avg_accuracy

def getLossAndGradNorm(model, trainset, batch_size, device):
    total_loss = 0
    grad_norm = 0
    #print("~~ trainset: ", type(trainset))

    one_inv_samples = torch.Tensor([1.0/len(trainset)]).to(device)

    dataloader = DataLoader(
                trainset,                  # dataset from which to load the data.
                batch_size=batch_size,     # How many samples per batch to load (default: 1).
                shuffle=False,             # Set to True to have the data reshuffled at every epoch (default: False)
                drop_last=False,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                pin_memory=False,          # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                collate_fn=None,           # Merges a list of samples to form a mini-batch of Tensor(s)
            )

    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)

    for p in model.parameters():
        p.grad = None

    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)                             # move to device

        logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network

        #print("~~logits", logits.shape)
        #print("~~outputs",outputs.shape)

        loss = one_inv_samples * F.cross_entropy(logits, outputs, reduction='sum')          # compute objective
        loss.backward()                                                                     # compute the gradient (backward-pass)
        total_loss += loss

    for p in model.parameters(): 
        grad_norm += torch.norm(p.grad.data.flatten(0))**2
        p.grad = None

    model.train(prev_train_mode)
    return total_loss, grad_norm

#======================================================================================================================================

class WorkerThreadDiana(threading.Thread):
  def __init__(self, wcfg, ncfg):
    threading.Thread.__init__(self)
    self.wcfg = wcfg
    self.ncfg = ncfg

    self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
    self.model = self.model.to(wcfg.device)                    # move model to device
    wcfg.model = self.model

    utils.setupAllParamsRandomly(self.model)
 
  def run(self):
    wcfg = self.wcfg
    ncfg = self.ncfg

    global transfered_bits_by_node
    global fi_grad_calcs_by_node
    global train_loss
    global test_loss
    global fn_train_loss_grad_norm
    global fn_test_loss_grad_norm

    # wcfg - configuration specific for worker
    # ncfg - general configuration with task description

    dbgprint(wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ", len(wcfg.train_set))
    #await init_workers_permission_event.wait()

    model = self.model

    loaders = (wcfg.train_loader, wcfg.test_loader)  # train and test loaders

    # Setup unitial shifts
    #=========================================================================================
    yk = utils.getAllParams(model)
    hk = zero_params_like(yk)
    #========================================================================================
    # Extra constants
    #one_div_trainset_all_len = torch.Tensor([1.0/len(wcfg.train_set_full)]).to(wcfg.device)
    one_div_trainset_len    = torch.Tensor([1.0/len(wcfg.train_set)]).to(wcfg.device)
    one_div_batch_prime_len = torch.Tensor([1.0/(ncfg.batch_size_for_worker)]).to(wcfg.device)
    delta_flatten = torch.zeros(ncfg.D).to(wcfg.device)
    #=========================================================================================
    iteration = 0
    #=========================================================================================
    full_grad_w = []
    #=========================================================================================
    while True:
        wcfg.input_cmd_ready.acquire()
        if wcfg.cmd == "exit":
            wcfg.output_of_cmd = ""
            wcfg.cmd_output_ready.release()
            break

        if wcfg.cmd == "bcast_xk_uk_0" or wcfg.cmd == "bcast_xk_uk_1":
            ts = [time()]

            # setup xk
            wcfg.output_of_cmd = []
            #================================================================================================================================
            # Generate subsample with b' cardinality
            if ncfg.i_use_vr_marina:
                indicies = torch.randperm(len(wcfg.train_set))[0:ncfg.batch_size_for_worker]
                subset = torch.utils.data.Subset(wcfg.train_set, indicies)
            else:
                subset = wcfg.train_set

            dbgprint(self.wcfg, 'WORKER USING SUBSET ', len(subset))

            ts.append(time())

            #================================================================================================================================
            minibatch_loader = DataLoader(
                subset,                    # dataset from which to load the data.
                batch_size=ncfg.batch_size,# How many samples per batch to load (default: 1).
                shuffle=False,             # Set to True to have the data reshuffled at every epoch (default: False)
                drop_last=False,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                pin_memory=False,          # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                collate_fn=None,           # Merges a list of samples to form a mini-batch of Tensor(s)
            )

            ts.append(time())

            #================================================================================================================================
            # Evaluate in current "xk" within b' batch 
            prev_train_mode = torch.is_grad_enabled()  
            #================================================================================================================================
            model.train(True)
            i = 0
            for p in model.parameters():
                p.data.flatten(0)[:] = wcfg.input_for_cmd[i].flatten(0)
                i += 1

            ts.append(time())

            for inputs, outputs in minibatch_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   # move to device
                logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
                loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs, reduction='sum')     # compute objective
                loss.backward()         
                                                            # compute the gradient (backward-pass)

            ts.append(time())

            gk_x = []
            for p in model.parameters():
                gk_x.append(p.grad.data.detach().clone())
                p.grad = None


            ts.append(time())

            #================================================================================================================================
            i = 0
            for p in model.parameters():
                p.data.flatten(0)[:] = yk[i].flatten(0)
                i += 1


            ts.append(time())

            for inputs, outputs in minibatch_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   # move to device
                logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
                loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs, reduction='sum')  # compute objective
                loss.backward()         # compute the gradient (backward-pass)


            ts.append(time())

            gk_w = []
            for p in model.parameters():
                gk_w.append(p.grad.data.detach().clone())
                p.grad = None
            #================================================================================================================================


            ts.append(time())

            for inputs, outputs in wcfg.train_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   # move to device
                logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
                loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')     # compute objective
                loss.backward()         

            if len(full_grad_w) == 0:                                                               # compute the gradient (backward-pass)
                for p in model.parameters():
                    full_grad_w.append(p.grad.data.detach().clone())
                    p.grad = None


            ts.append(time())

            #================================================================================================================================
            model.train(prev_train_mode)
            #================================================================================================================================
            #dbgprint(wcfg, "gkx", len(gk_x))
            #dbgprint(wcfg, "gkw", len(gk_w))
            #dbgprint(wcfg, "gfull", len(full_grad_w))
            #dbgprint(wcfg, "hk", hk)

            gk_next = add_params(sub_params(gk_x, gk_w), full_grad_w)
            delta   = sub_params(gk_next, hk)

            ts.append(time())

            # Compress delta
            #================================================================================================================================
            delta_offset = 0
            for t in range(len(delta)):
                offset = len(delta[t].flatten(0))
                delta_flatten[(delta_offset):(delta_offset + offset)] = delta[t].flatten(0)
                delta_offset += offset

            delta_flatten = wcfg.compressor.compressVector(delta_flatten)             # Compress shifted local gradient

            delta_offset = 0
            for t in range(len(delta)):
                offset = len(delta[t].flatten(0))
                delta[t].flatten(0)[:] = delta_flatten[(delta_offset):(delta_offset + offset)]
                delta_offset += offset
            #================================================================================================================================
            mk_i = delta
            hk = add_params(hk, mult_param(nn_config.fixed_alpha_diana, mk_i))
            #================================================================================================================================
            i = 0
            for p in model.parameters(): 
                wcfg.output_of_cmd.append(mk_i[i].data.detach().clone())
                i += 1
            #================================================================================================================================
            transfered_bits_by_node[wcfg.worker_id, iteration] = wcfg.compressor.last_need_to_send_advance * ncfg.component_bits_size

            if iteration == 0:
                # need to take full grad at beginning
                fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(wcfg.train_set)
            elif wcfg.cmd == "bcast_xk_uk_1":
                # update control imply more big gradient at next step
                fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(wcfg.train_set) + 2 * ncfg.batch_size_for_worker
            else:
                fi_grad_calcs_by_node[wcfg.worker_id, iteration] = 2 * ncfg.batch_size_for_worker

            if wcfg.cmd == "bcast_xk_uk_1":  
                for i in range(len(yk)):
                    yk[i].flatten(0)[:] = wcfg.input_for_cmd[i].flatten(0)
                full_grad_w = []

            iteration += 1                       
            #================================================================================================================================


            ts.append(time())
            dbgprint(self.wcfg, [ts[i + 1] - ts[i] for i in range(len(ts) - 1)])

            wcfg.cmd_output_ready.release()
        #===========================================================================================

    # Signal that worker has finished initialization via decreasing semaphore
    #completed_workers_semaphore.acquire()
    dbgprint(wcfg, f"END")
#======================================================================================================================================

class WorkerThreadMarina(threading.Thread):
    def __init__(self, wcfg, ncfg):
        threading.Thread.__init__(self)
        self.wcfg = wcfg
        self.ncfg = ncfg

        self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
        self.model = self.model.to(wcfg.device)  # move model to device
        utils.setupAllParamsRandomly(self.model)
        wcfg.model = self.model

    def run(self):
        wcfg = self.wcfg
        ncfg = self.ncfg

        global transfered_bits_by_node
        global fi_grad_calcs_by_node
        global train_loss
        global test_loss
        global fn_train_loss_grad_norm
        global fn_test_loss_grad_norm

        # wcfg - configuration specific for worker
        # ncfg - general configuration with task description

        dbgprint(wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ",
                 len(wcfg.train_set))
        # await init_workers_permission_event.wait()

        model = self.model

        loaders = (wcfg.train_loader, wcfg.test_loader)  # train and test loaders

        # one_div_trainset_all_len = torch.Tensor([1.0/len(wcfg.train_set_full)]).to(wcfg.device)
        one_div_trainset_len = torch.Tensor([1.0 / len(wcfg.train_set)]).to(wcfg.device)

        if ncfg.i_use_vr_marina:
            one_div_batch_prime_len = torch.Tensor([1.0 / (ncfg.batch_size_for_worker)]).to(wcfg.device)
        else:
            one_div_batch_prime_len = torch.Tensor([1.0 / len(wcfg.train_set)]).to(wcfg.device)

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

            #        if wcfg.cmd == "curr_x":
            #            wcfg.output_of_cmd = []
            #            for p in model.parameters():
            #                wcfg.output_of_cmd.append(p.data.detach().clone())
            #            wcfg.cmd_output_ready.release()

            # dbgprint(self.wcfg, wcfg.cmd)

            if wcfg.cmd == "bcast_g_c1":
                wcfg.output_of_cmd = []
                k = 0
                for p in model.parameters():
                    p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[k]
                    k = k + 1

                prev_train_mode = torch.is_grad_enabled()
                model.train(True)

                for inputs, outputs in wcfg.train_loader:
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')  # compute objective
                    loss.backward()  # compute the gradient (backward-pass)

                for p in model.parameters():
                    wcfg.output_of_cmd.append(p.grad.data.detach().clone())
                    p.grad = None

                model.train(prev_train_mode)

                # Case when we send to master really complete gradient
                transfered_bits_by_node[wcfg.worker_id, iteration] = ncfg.D * ncfg.component_bits_size
                # On that "c1" mode to evaluate gradient we call oracle number of times how much data is in local "full" batch
                fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(wcfg.train_set)

                wcfg.cmd_output_ready.release()
                iteration += 1

            if wcfg.cmd == "bcast_g_c0":
                wcfg.output_of_cmd = []

                # ================================================================================================================================
                # Generate subsample with b' cardinality
                indicies = None
                if ncfg.i_use_vr_marina:
                    indicies = torch.randperm(len(wcfg.train_set))[0:ncfg.batch_size_for_worker]
                    subset = torch.utils.data.Subset(wcfg.train_set, indicies)
                else:
                    subset = wcfg.train_set
                # ================================================================================================================================
                minibatch_loader = DataLoader(
                    subset,  # dataset from which to load the data.
                    batch_size=ncfg.batch_size,  # How many samples per batch to load (default: 1).
                    shuffle=False,  # Set to True to have the data reshuffled at every epoch (default: False)
                    drop_last=False,
                    # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                    pin_memory=False,
                    # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                    collate_fn=None,  # Merges a list of samples to form a mini-batch of Tensor(s)
                )

                # Evaluate in previous point SGD within b' batch
                prev_train_mode = torch.is_grad_enabled()
                model.train(True)

                # ================================================================================================================================
                from time import time
                t0 = time()
                for t, (inputs, outputs) in enumerate(minibatch_loader):
                    t1 = time()
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs,
                                                                     reduction='sum')  # compute objective
                    loss.backward()  # compute the gradient (backward-pass)
                    t5 = time()
                    # dbgprint(self.wcfg, f'For MB cat: {t1 - t0:.3f}; For rest: {t5 - t1:.3f}')

                g_batch_prev = []
                for p in model.parameters():
                    g_batch_prev.append(p.grad.data.detach().clone())
                    p.grad = None
                # ================================================================================================================================
                # Change xk: move from x(k) to x(k+1)
                k = 0
                for p in model.parameters():
                    p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[k]
                    k = k + 1

                # Evaluate SGD in the new point within b' batch
                # ================================================================================================================================
                t0 = time()
                for inputs, outputs in minibatch_loader:
                    t1 = time()
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs,
                                                                     reduction='sum')  # compute objective
                    loss.backward()  # compute the gradient (backward-pass)
                    t5 = time()
                    # dbgprint(self.wcfg, f'For MB cat: {t1 - t0:.3f}; For rest: {t5 - t1:.3f}')

                t0 = time()
                g_batch_next = []
                for p in model.parameters():
                    g_batch_next.append(p.grad.data.detach().clone())
                    p.grad = None
                # dbgprint(self.wcfg, f'For Grad clone -> g_batch_next: {time() - t0:.3f}')

                # t0 = time() # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                delta = sub_params(g_batch_next, g_batch_prev)

                t1 = time()
                # dbgprint(self.wcfg, f'For sub params: {t1 - t0:.3f}')

                delta_offset = 0
                for t in range(len(delta)):
                    offset = len(delta[t].flatten(0))
                    delta_flatten[(delta_offset):(delta_offset + offset)] = delta[t].flatten(0)
                    delta_offset += offset

                # dbgprint(self.wcfg, f'Delta offsetting: {time() - t1:.3f}')
                # t1 = time()

                delta_flatten = wcfg.compressor.compressVector(delta_flatten)
                # dbgprint(self.wcfg, f'Compressing delta_flatten: {time() - t1:.3f}')
                # t1 = time()

                delta_offset = 0
                for t in range(len(delta)):
                    offset = len(delta[t].flatten(0))
                    delta[t].flatten(0)[:] = delta_flatten[(delta_offset):(delta_offset + offset)]
                    delta_offset += offset

                # dbgprint(self.wcfg, f'Delta offsetting p2: {time() - t1:.3f}')
                # t1 = time()

                g_new = add_params(wcfg.input_for_cmd, delta)
                wcfg.output_of_cmd = g_new
                model.train(prev_train_mode)

                transfered_bits_by_node[
                    wcfg.worker_id, iteration] = wcfg.compressor.last_need_to_send_advance * ncfg.component_bits_size
                fi_grad_calcs_by_node[wcfg.worker_id, iteration] = 2 * ncfg.batch_size_for_worker
                iteration += 1

                wcfg.cmd_output_ready.release()

                # dbgprint(self.wcfg, f'For end of work: {time() - t1:.3f}')

            if wcfg.cmd == "full_grad":
                wcfg.output_of_cmd = []
                prev_train_mode = torch.is_grad_enabled()
                model.train(True)

                for inputs, outputs in wcfg.train_loader:
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_trainset_len * F.cross_entropy(logits, outputs)  # compute objective
                    loss.backward()  # compute the gradient (backward-pass)

                for p in model.parameters():
                    wcfg.output_of_cmd.append(p.grad.data.detach().clone())
                    p.grad = None

                model.train(prev_train_mode)
                wcfg.cmd_output_ready.release()

        # Signal that worker has finished initialization via decreasing semaphore
        # completed_workers_semaphore.acquire()
        dbgprint(wcfg, f"END")

# =====================================================================================================================




def main(algo_name=AlgoNames.VR_MARINA,
         batch_size_for_worker=256, technical_batch_size=512,
         gamma=1e-3, K=100_000, n_iters=3500, no_compr=False, p=None,
         log_every=50, device_num=5, debug=False):

    global transfered_bits_by_node
    global fi_grad_calcs_by_node
    global train_loss
    global test_loss
    global train_acc
    global test_acc
    global fn_train_loss_grad_norm
    global fn_test_loss_grad_norm
    global nn_config, workers_config

    nn_config = NNConfiguration()

    nn_config.i_use_marina = False
    nn_config.i_use_vr_marina = False
    nn_config.i_use_diana = False
    nn_config.i_use_vr_diana = False

    if algo_name == AlgoNames.DIANA:
        nn_config.i_use_diana = True
    if algo_name == AlgoNames.VR_DIANA:
        nn_config.i_use_vr_diana = True
    if algo_name == AlgoNames.MARINA:
        nn_config.i_use_marina = True
    if algo_name == AlgoNames.VR_MARINA:
        nn_config.i_use_vr_marina = True

    # t_start = time()
    # print(f'Start TIME {t_start}')
    # utils.printTorchInfo()
    torch.manual_seed(1)        # Set the random seed so things involved torch.randn are predictable/repetable


    cpu_device = torch.device("cpu")  # CPU device
    gpu_devices = [torch.device(f'cuda:{i}') for i in range(8)]
    available_devices = [gpu_devices[device_num]]
    master_device = available_devices[0]

    # Configuration for NN

    nn_config.dataset = "CIFAR100"          # Dataset
    nn_config.model_name = "resnet18"      # NN architecture
    nn_config.load_workers = 0             # KEEP THIS 0. How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    nn_config.batch_size = technical_batch_size             # Technical batch size for training (due to GPU limitations)
    nn_config.KMax = n_iters                  # Maximum number of iterations
    nn_config.K = K
    nn_config.algo_name = algo_name
    kWorkers = 5 if not debug else 1
    nn_config.kWorkers = kWorkers  # Number of workers

    #=======================================================================================================
    # Load data
    t1 = time()
    train_sets, train_set_full, test_set, train_loaders, test_loaders, classes = utils.getSplitDatasets(nn_config.dataset, nn_config.batch_size,
                                                                                                        nn_config.load_workers, kWorkers,
                                                                                                        ram=True, ram_device=available_devices[0],
                                                                                                        debug=debug)
    print(f'GetSplitDatasets took {time() - t1:.3f}')

    # Load model
    t1 = time()
    master_model = utils.getModel(nn_config.model_name, train_set_full, master_device)
    print(f'getModel took {time() - t1:.3f}')
    # utils.printLayersInfo(master_model, nn_config.model_name)
    #=======================================================================================================

    nn_config.train_set_full_samples = len(train_set_full)
    nn_config.train_sets_samples = [len(s) for s in train_sets]
    nn_config.test_set_samples = len(test_set)
    nn_config.D = utils.numberOfParams(master_model)
    nn_config.component_bits_size = 32

    # Initialize "xk <- x0"
    utils.setupAllParamsRandomly(master_model)
    xk = utils.getAllParams(master_model)
    hk = zero_params_like(xk)

    # Statistics/Metrics during training
    transfered_bits_by_node = np.zeros((kWorkers, nn_config.KMax)) # Transfered bits
    fi_grad_calcs_by_node   = np.zeros((kWorkers, nn_config.KMax)) # Evaluate number gradients for fi
    train_loss              = np.zeros((nn_config.KMax))           # Train loss
    test_loss               = np.zeros((nn_config.KMax))           # Validation loss
    train_acc               = np.zeros((nn_config.KMax))           # Train accuracy
    test_acc                = np.zeros((nn_config.KMax))           # Validation accuracy

    fn_train_loss_grad_norm = np.zeros((nn_config.KMax))           # Gradient norm for train loss
    fn_test_loss_grad_norm  = np.zeros((nn_config.KMax))           # Gradient norm for test loss


    # Create Compressor =======================================================
    c = compressors.Compressor()
    if no_compr:
        c.makeIdenticalCompressor()
    else:
        c.makeRandKCompressor(int(K), nn_config.D)
    w = c.getW()
    # =========================================================================

    # Set params for each algorithm separately
    if algo_name == AlgoNames.DIANA:
      nn_config.gamma = gamma                                    # Gamma for DIANA
      nn_config.fixed_alpha_diana = 1.0/(1+w)                    # Alpha for DIANA
    elif algo_name == AlgoNames.VR_DIANA:
      nn_config.gamma = gamma                                    # Gamma for VR-DIANA
      nn_config.fixed_alpha_diana = 1.0/(1+w)                    # Alpha for VR-DIANA
      nn_config.batch_size_for_worker = batch_size_for_worker    # Batch for VR-DIANA
      m = (len(train_sets[0]))/(nn_config.batch_size_for_worker)
      nn_config.p     = 1.0/m                   # p for VR-DIANA
    elif algo_name == AlgoNames.VR_MARINA:
        nn_config.gamma = gamma  # Gamma for VR-MARINA
        nn_config.batch_size_for_worker = batch_size_for_worker # Batch for VR-MARINA
        if p is None:
            nn_config.p = min(1.0 / (1 + w),
                              (nn_config.batch_size_for_worker) / (nn_config.batch_size_for_worker + len(train_sets[0])))
        else:
            nn_config.p = p

    # Creater worker threads
    t1 = time()

    worker_tasks = []                           # Worker tasks
    worker_cfgs = []                            # Worker configurations

    for i in range(kWorkers):
        worker_cfgs.append(WorkersConfiguration())
        worker_cfgs[-1].worker_id = i
        worker_cfgs[-1].total_workers = kWorkers
        worker_cfgs[-1].train_set = train_sets[i]
        worker_cfgs[-1].test_set = test_set
        worker_cfgs[-1].train_set_full = train_set_full

        worker_cfgs[-1].train_loader = train_loaders[i]
        worker_cfgs[-1].test_loader = test_loaders[i]
        worker_cfgs[-1].classes = classes
        worker_cfgs[-1].device = available_devices[i % len(available_devices)]                 # device used by worker
        worker_cfgs[-1].compressor = compressors.Compressor()
        if no_compr:
            worker_cfgs[-1].compressor.makeIdenticalCompressor()
        else:
            worker_cfgs[-1].compressor.makeRandKCompressor(int(K), nn_config.D)

        worker_cfgs[-1].input_cmd_ready  = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd_output_ready = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd = "init"
        worker_cfgs[-1].input_for_cmd = ""
        worker_cfgs[-1].output_of_cmd = ""

        if algo_name == AlgoNames.DIANA or algo_name == AlgoNames.VR_DIANA:
            worker_tasks.append(WorkerThreadDiana(worker_cfgs[-1], nn_config))
        else:
            worker_tasks.append(WorkerThreadMarina(worker_cfgs[-1], nn_config))


    # Start worker threads
    for i in range(kWorkers):
        worker_tasks[i].start()

    rootprint(f'Starting workers took {time() - t1:.3f}')
    rootprint(f"Start training {nn_config.model_name}@{nn_config.dataset} for K={nn_config.KMax} iteration")

    #===================================================================================
    if algo_name == AlgoNames.VR_DIANA:
        for iteration in range(0, nn_config.KMax):
            #if k % 2 == 0:
            rootprint(f"Iteration {iteration}/{nn_config.KMax}. Completed by ", iteration/nn_config.KMax * 100.0, "%")

            #====================================================================
            # Collect statistics
            #====================================================================
            if iteration % log_every == 0:
                utils.setupAllParams(master_model, xk)

                loss, grad_norm = getLossAndGradNorm(master_model, train_set_full, nn_config.batch_size, master_device)
                train_loss[iteration] = loss
                fn_train_loss_grad_norm[iteration] = grad_norm

                loss, grad_norm = getLossAndGradNorm(master_model, test_set, nn_config.batch_size, master_device)
                test_loss[iteration] = loss
                fn_test_loss_grad_norm[iteration] = grad_norm

                train_acc[iteration] = getAccuracy(master_model, train_set_full, nn_config.batch_size, master_device)
                test_acc[iteration]  = getAccuracy(master_model, test_set, nn_config.batch_size, master_device)
                print(f"  train accuracy: {train_acc[iteration]}, test accuracy: {test_acc[iteration]}, train loss: {train_loss[iteration]}, test loss: {test_loss[iteration]}")
                print(f"  grad norm train: {fn_train_loss_grad_norm[iteration]}, test: {fn_test_loss_grad_norm[iteration]}")
                print(f"  used step-size: {nn_config.gamma}")

            else:
                train_loss[iteration]              = train_loss[iteration - 1] 
                fn_train_loss_grad_norm[iteration] = fn_train_loss_grad_norm[iteration - 1]
                test_loss[iteration]               = test_loss[iteration - 1]
                fn_test_loss_grad_norm[iteration]  = fn_test_loss_grad_norm[iteration - 1] 
                train_acc[iteration]               = train_acc[iteration - 1]
                test_acc[iteration]                = test_acc[iteration - 1]

            #====================================================================

            # Draw testp Bernoulli random variable (which is equal 1 w.p. p)
            uk = 0
            testp = random.random()
            if testp < nn_config.p:
                uk = 1
            else:
                uk = 0

            xk_for_device = {}
            for d_id in range(len(available_devices)):
                xk_loc = []
                for xk_i in xk:
                    xk_loc.append(xk_i.to(available_devices[d_id]))
                xk_for_device[available_devices[d_id]] = xk_loc

            #===========================================================================           
            # Generate control
            #===========================================================================

            t0 = time()
            if uk == 1:
                # Broadcast xk and obtain gi as reponse from workers
                for i in range(kWorkers):
                    worker_cfgs[i].cmd = "bcast_xk_uk_1"
                    worker_cfgs[i].input_for_cmd = xk_for_device[worker_cfgs[i].device]
                    worker_cfgs[i].input_cmd_ready.release()
                for i in range(kWorkers):
                    worker_cfgs[i].cmd_output_ready.acquire()
            elif uk == 0:
                # Broadcast xk and obtain gi as reponse from workers
                for i in range(kWorkers):
                    worker_cfgs[i].cmd = "bcast_xk_uk_0"
                    worker_cfgs[i].input_for_cmd = xk_for_device[worker_cfgs[i].device]
                    worker_cfgs[i].input_cmd_ready.release()
                for i in range(kWorkers):
                    worker_cfgs[i].cmd_output_ready.acquire()
            print(f'Waiting for workers: {time() - t0:.3f}')

            #===========================================================================           
            # Aggregate received messages (From paper: (delta^)->gk)
            #===========================================================================
            mk_avg = worker_cfgs[0].output_of_cmd
            worker_cfgs[0].output_of_cmd = None
            for i in range(1, kWorkers): 
                for j in range(len(worker_cfgs[i].output_of_cmd)):
                    mk_avg[j] = mk_avg[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
                worker_cfgs[i].output_of_cmd = None
            mk_avg = mult_param(1.0/kWorkers, mk_avg)

            #===========================================================================           
            # Need updates on master node
            #===========================================================================
            # Compute global gradient estimator:
            gk = add_params(hk, mk_avg)
            # Take proximal SGD step
            xk = sub_params(xk, mult_param(nn_config.gamma, gk))
            # Update aggregated shift:
            hk = add_params(hk, mult_param(nn_config.fixed_alpha_diana, mk_avg))

    elif algo_name == AlgoNames.MARINA or algo_name == AlgoNames.VR_MARINA:
        # Evaluate g0
        for i in range(kWorkers):
            worker_cfgs[i].cmd = "full_grad"
            worker_cfgs[i].input_cmd_ready.release()

        for i in range(kWorkers):
            worker_cfgs[i].cmd_output_ready.acquire()

        g0 = worker_cfgs[0].output_of_cmd
        worker_cfgs[0].output_of_cmd = None
        for i in range(1, kWorkers):
            for j in range(len(worker_cfgs[i].output_of_cmd)):
                g0[j] = g0[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
            worker_cfgs[i].output_of_cmd = None
        g0 = mult_param(1.0 / kWorkers, g0)
        gk = g0

        rootprint(f"Start {nn_config.KMax} iterations of algorithm")

        prev_time = time()
        for iteration in range(0, nn_config.KMax):
            # if k % 2 == 0:
            rootprint(f"Iteration {iteration}/{nn_config.KMax}. Completed by ", iteration / nn_config.KMax * 100.0, "%",
                      f'elapsed {time() - prev_time} s')
            prev_time = time()
            # ====================================================================
            # Collect statistics
            # ====================================================================
            if iteration % log_every == 0:
                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, train_set_full, nn_config.batch_size,
                                                     worker_cfgs[0].device)
                train_loss[iteration] = loss
                fn_train_loss_grad_norm[iteration] = grad_norm

                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, test_set, nn_config.batch_size,
                                                     worker_cfgs[0].device)
                test_loss[iteration] = loss
                fn_test_loss_grad_norm[iteration] = grad_norm

                train_acc[iteration] = getAccuracy(worker_cfgs[0].model, train_set_full, nn_config.batch_size,
                                                   worker_cfgs[0].device)
                test_acc[iteration] = getAccuracy(worker_cfgs[0].model, test_set, nn_config.batch_size,
                                                  worker_cfgs[0].device)
                print(
                    f"  train accuracy: {train_acc[iteration]}, test accuracy: {test_acc[iteration]}, train loss: {train_loss[iteration]}, test loss: {test_loss[iteration]}")
                print(
                    f"  grad norm train: {fn_train_loss_grad_norm[iteration]}, test: {fn_test_loss_grad_norm[iteration]}")
                print(f"  used step-size: {nn_config.gamma}")

            else:
                train_loss[iteration] = train_loss[iteration - 1]
                fn_train_loss_grad_norm[iteration] = fn_train_loss_grad_norm[iteration - 1]
                test_loss[iteration] = test_loss[iteration - 1]
                fn_test_loss_grad_norm[iteration] = fn_test_loss_grad_norm[iteration - 1]
                train_acc[iteration] = train_acc[iteration - 1]
                test_acc[iteration] = test_acc[iteration - 1]
            # ====================================================================

            # Draw testp Bernoulli random variable (which is equal 1 w.p. p)
            ck = 0
            testp = random.random()
            if testp < nn_config.p:
                ck = 1
            else:
                ck = 0

            # Broadcast gk wti command to workers
            gk_for_device = {}
            for d_id in range(len(available_devices)):
                gk_loc = []
                for gk_i in gk:
                    gk_loc.append(gk_i.to(available_devices[d_id]))
                gk_for_device[available_devices[d_id]] = gk_loc

            if ck == 1:
                for i in range(kWorkers):
                    worker_cfgs[i].cmd = "bcast_g_c1"
                    worker_cfgs[i].input_for_cmd = gk_for_device[worker_cfgs[i].device]
                    worker_cfgs[i].input_cmd_ready.release()

            if ck == 0:
                for i in range(kWorkers):
                    worker_cfgs[i].cmd = "bcast_g_c0"
                    worker_cfgs[i].input_for_cmd = gk_for_device[worker_cfgs[i].device]
                    worker_cfgs[i].input_cmd_ready.release()

            time_before = time()
            # rootprint(f'Time spent BEFORE waiting for workers: {time_before - prev_time}')
            # Obtain workers gi and average result
            for i in range(kWorkers):
                worker_cfgs[i].cmd_output_ready.acquire()
            time_after = time()
            rootprint(f'Time spent waiting for workers: {time_after - time_before}')

            gk_next = worker_cfgs[0].output_of_cmd
            worker_cfgs[0].output_of_cmd = None
            for i in range(1, kWorkers):
                for j in range(len(worker_cfgs[i].output_of_cmd)):
                    gk_next[j] = gk_next[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
                worker_cfgs[i].output_of_cmd = None
            gk_next = mult_param(1.0 / kWorkers, gk_next)
            print('Norm of change: ', norm_of_param(sub_params(gk, gk_next)))
            gk = gk_next

            # rootprint(f'Time spent AFTER waiting for workers: {time() - time_after}')

    #===================================================================================

    # Finish all work of nodes
    for i in range(kWorkers):
        worker_cfgs[i].cmd = "exit"
        worker_cfgs[i].input_cmd_ready.release()
    #==================================================================================
    for i in range(kWorkers):
        worker_tasks[i].join()
    print(f"Master has been finished")
    #==================================================================================

    # Serialize statistices & description
    my = {}
    my["transfered_bits_by_node"] = transfered_bits_by_node
    my["fi_grad_calcs_by_node"] = fi_grad_calcs_by_node

    my["train_loss"] = train_loss
    my["test_loss"] = test_loss
    my["train_acc"] = train_acc
    my["test_acc"]  = test_acc

    my["fn_train_loss_grad_norm"] = fn_train_loss_grad_norm
    my["fn_test_loss_grad_norm"] = fn_test_loss_grad_norm
    my["nn_config"] = nn_config
    my["current_data_and_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    my["experiment_description"] = f"Training {nn_config.model_name}@{nn_config.dataset}"
    my["compressors"] = worker_cfgs[0].compressor.fullName()
    my["algo_name"] = f"{algo_name}"
    if hasattr(worker_cfgs[0].compressor, "K"):
        my["compressors_rand_K"] = worker_cfgs[0].compressor.K
    else:
        my["compressors_rand_K"] = None

    log_file_name = utils.create_log_file(f'{algo_name} \n'
                                          f'niters={n_iters} \n'
                                          f'K={K} \n'
                                          f'nocompr={no_compr} \n'
                                          f'gamma={gamma} \n'
                                          f'p={p} \n'
                                          f'batch_size_for_worker={batch_size_for_worker} \n'
                                          f'technical_batch_size={technical_batch_size} \n'
                                          f'nocompr={no_compr} \n'
                                          )

    print('SERIALIZING: ', log_file_name)
    print('SERIALIZING: ', my)

    utils.serialize(my, log_file_name)

    print(f"Experiment info has been serialised into '{log_file_name}'")
    #==================================================================================

if __name__ == "__main__":
    DEBUG = True
    device_num = 1

    n_iter = 10
    log_every = 10

    alg_names = []
    n_iters = []
    Ks = []
    no_comprs = []
    batch_size_for_worker = []
    technical_batch_size = []
    gamma = []
    ps = []

    # alg_names += [AlgoNames.VR_DIANA] * 4
    # n_iters += [n_iter] * 4
    # Ks += [100_000, 500_000, 1_000_000, None]
    # no_comprs += [False, False, False, True]
    # batch_size_for_worker += [256 * 1] * 4
    # technical_batch_size += [256 * 16] * 4
    # gamma += ([0.95] * 3 + [3.5])
    # ps += [None] * 4

    alg_names += [AlgoNames.VR_DIANA] * 4
    n_iters += [n_iter] * 4
    Ks += [100_000, 500_000, 1_000_000, None]
    no_comprs += [False, False, False, True]
    batch_size_for_worker += [256 * 1] * 4
    technical_batch_size += [256 * 16] * 4
    gamma +=  [0.15, 0.35, 0.35, 2.5]
    ps += [None] * 4

    alg_names += [AlgoNames.VR_MARINA] * 5
    n_iters += [n_iter] * 5
    Ks += [100_000, 500_000, 1_000_000, None, None]
    no_comprs += [False, False, False, True, True]
    batch_size_for_worker += [256 * 1] * 5
    technical_batch_size += [256 * 16] * 5
    gamma += [0.95] * 3 + [3.5] * 2
    ps += [None] * 4 + [0.008554677047253982]

    iter_id = 0
    do_iters = list(range(len(n_iters)))
    for conf in zip(alg_names, n_iters, Ks, no_comprs, batch_size_for_worker, technical_batch_size, gamma, ps):
        a, n, K, nc, bw, tb, g, p = conf
        if iter_id in do_iters:
            main(algo_name=a, n_iters=n,
                 K=K, no_compr=nc, gamma=g, p=p, batch_size_for_worker=bw,
                 technical_batch_size=tb, device_num=device_num,
                 debug=DEBUG,
                 log_every=log_every)
        iter_id += 1
