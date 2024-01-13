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
from debug_utils import rootprint, dbgprint

# Worker thread implementations for each algorithm
from WorkerThreadDiana import WorkerThreadDiana
from WorkerThreadMarina import WorkerThreadMarina


#====================================================================================
class NNConfiguration: pass
class WorkersConfiguration: pass

class AlgoNames:
    DIANA = 0
    VR_DIANA = 1
    MARINA = 2
    VR_MARINA = 3
    KATYUSHA = 4

#====================================================================================


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



def main(algo_name=AlgoNames.VR_MARINA,
         batch_size_for_worker=256, technical_batch_size=512,
         gamma=1e-3, K=100_000, n_iters=3500, no_compr=False, p=None,
         log_every=50, device_num=5, debug=False):


    cpu_device = torch.device("cpu")  # CPU device
    gpu_devices = [torch.device(f'cuda:{i}') for i in range(8)]
    available_devices = [gpu_devices[device_num]]
    master_device = available_devices[0]
    torch.manual_seed(1)

    print_lock = threading.Lock()

    # ================================================================================

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

    # Configuration for NN

    nn_config.dataset = "CIFAR100"          # Dataset
    nn_config.model_name = "resnet18"      # NN architecture
    nn_config.load_workers = 0             # KEEP THIS 0. How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    nn_config.technical_batch_size = technical_batch_size             # Technical batch size for training (due to GPU limitations)
    nn_config.KMax = n_iters                  # Maximum number of iterations
    nn_config.K = K
    nn_config.algo_name = algo_name
    kWorkers = 5 if not debug else 1
    nn_config.kWorkers = kWorkers  # Number of workers

    #=======================================================================================================
    # Load data
    t1 = time()
    train_sets, train_set_full, test_set, train_loaders, test_loaders, classes = utils.getSplitDatasets(nn_config.dataset, nn_config.technical_batch_size,
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

    stats = NNConfiguration()
    stats.transfered_bits_by_node = transfered_bits_by_node
    stats.fi_grad_calcs_by_node = fi_grad_calcs_by_node
    stats.train_loss = train_loss
    stats.test_loss = test_loss
    stats.train_acc = train_acc
    stats.test_acc = test_acc
    stats.fn_train_loss_grad_norm = fn_train_loss_grad_norm
    stats.fn_test_loss_grad_norm = fn_test_loss_grad_norm

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
            worker_tasks.append(WorkerThreadDiana(worker_cfgs[-1], nn_config, stats, print_lock=print_lock))
        elif algo_name == AlgoNames.MARINA or algo_name == AlgoNames.VR_MARINA:
            worker_tasks.append(WorkerThreadMarina(worker_cfgs[-1], nn_config, stats, print_lock=print_lock))

    # Start worker threads
    for i in range(kWorkers):
        worker_tasks[i].start()

    rootprint(print_lock, f'Starting workers took {time() - t1:.3f}')
    rootprint(print_lock, f"Start training {nn_config.model_name}@{nn_config.dataset} for K={nn_config.KMax} iteration")

    #===================================================================================
    if algo_name == AlgoNames.VR_DIANA:
        for iteration in range(0, nn_config.KMax):
            #if k % 2 == 0:
            rootprint(print_lock, f"Iteration {iteration}/{nn_config.KMax}. Completed by ", iteration/nn_config.KMax * 100.0, "%")

            #====================================================================
            # Collect statistics
            #====================================================================
            if iteration % log_every == 0:
                utils.setupAllParams(master_model, xk)

                loss, grad_norm = getLossAndGradNorm(master_model, train_set_full, nn_config.technical_batch_size, master_device)
                train_loss[iteration] = loss
                fn_train_loss_grad_norm[iteration] = grad_norm

                loss, grad_norm = getLossAndGradNorm(master_model, test_set, nn_config.technical_batch_size, master_device)
                test_loss[iteration] = loss
                fn_test_loss_grad_norm[iteration] = grad_norm

                train_acc[iteration] = getAccuracy(master_model, train_set_full, nn_config.technical_batch_size, master_device)
                test_acc[iteration]  = getAccuracy(master_model, test_set, nn_config.technical_batch_size, master_device)
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
                # Broadcast xk and obtain gi as response from workers
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

        rootprint(print_lock, f"Start {nn_config.KMax} iterations of algorithm")

        prev_time = time()
        for iteration in range(0, nn_config.KMax):
            # if k % 2 == 0:
            rootprint(print_lock, f"Iteration {iteration}/{nn_config.KMax}. Completed by ", iteration / nn_config.KMax * 100.0, "%",
                      f'elapsed {time() - prev_time} s')
            prev_time = time()
            # ====================================================================
            # Collect statistics
            # ====================================================================
            if iteration % log_every == 0:
                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, train_set_full, nn_config.technical_batch_size,
                                                     worker_cfgs[0].device)
                train_loss[iteration] = loss
                fn_train_loss_grad_norm[iteration] = grad_norm

                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, test_set, nn_config.technical_batch_size,
                                                     worker_cfgs[0].device)
                test_loss[iteration] = loss
                fn_test_loss_grad_norm[iteration] = grad_norm

                train_acc[iteration] = getAccuracy(worker_cfgs[0].model, train_set_full, nn_config.technical_batch_size,
                                                   worker_cfgs[0].device)
                test_acc[iteration] = getAccuracy(worker_cfgs[0].model, test_set, nn_config.technical_batch_size,
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
            # rootprint(print_lock, f'Time spent BEFORE waiting for workers: {time_before - prev_time}')
            # Obtain workers gi and average result
            for i in range(kWorkers):
                worker_cfgs[i].cmd_output_ready.acquire()
            time_after = time()
            rootprint(print_lock, f'Time spent waiting for workers: {time_after - time_before}')

            gk_next = worker_cfgs[0].output_of_cmd
            worker_cfgs[0].output_of_cmd = None
            for i in range(1, kWorkers):
                for j in range(len(worker_cfgs[i].output_of_cmd)):
                    gk_next[j] = gk_next[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
                worker_cfgs[i].output_of_cmd = None
            gk_next = mult_param(1.0 / kWorkers, gk_next)
            print('Norm of change: ', norm_of_param(sub_params(gk, gk_next)))
            gk = gk_next

            # rootprint(print_lock, f'Time spent AFTER waiting for workers: {time() - time_after}')

    elif algo_name == AlgoNames.KATYUSHA:
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

        rootprint(print_lock, f"Start {nn_config.KMax} iterations of algorithm")

        prev_time = time()
        for iteration in range(0, nn_config.KMax):
            # if k % 2 == 0:
            rootprint(print_lock, f"Iteration {iteration}/{nn_config.KMax}. Completed by ",
                      iteration / nn_config.KMax * 100.0, "%",
                      f'elapsed {time() - prev_time} s')
            prev_time = time()
            # ====================================================================
            # Collect statistics
            # ====================================================================
            if iteration % log_every == 0:
                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, train_set_full,
                                                     nn_config.technical_batch_size,
                                                     worker_cfgs[0].device)
                train_loss[iteration] = loss
                fn_train_loss_grad_norm[iteration] = grad_norm

                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, test_set, nn_config.technical_batch_size,
                                                     worker_cfgs[0].device)
                test_loss[iteration] = loss
                fn_test_loss_grad_norm[iteration] = grad_norm

                train_acc[iteration] = getAccuracy(worker_cfgs[0].model, train_set_full, nn_config.technical_batch_size,
                                                   worker_cfgs[0].device)
                test_acc[iteration] = getAccuracy(worker_cfgs[0].model, test_set, nn_config.technical_batch_size,
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
            # rootprint(print_lock, f'Time spent BEFORE waiting for workers: {time_before - prev_time}')
            # Obtain workers gi and average result
            for i in range(kWorkers):
                worker_cfgs[i].cmd_output_ready.acquire()
            time_after = time()
            rootprint(print_lock, f'Time spent waiting for workers: {time_after - time_before}')

            gk_next = worker_cfgs[0].output_of_cmd
            worker_cfgs[0].output_of_cmd = None
            for i in range(1, kWorkers):
                for j in range(len(worker_cfgs[i].output_of_cmd)):
                    gk_next[j] = gk_next[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
                worker_cfgs[i].output_of_cmd = None
            gk_next = mult_param(1.0 / kWorkers, gk_next)
            print('Norm of change: ', norm_of_param(sub_params(gk, gk_next)))
            gk = gk_next

            # rootprint(print_lock, f'Time spent AFTER waiting for workers: {time() - time_after}')

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
    DEBUG = False
    device_num = 0

    n_iter = 500
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
    # do_iters = [4]
    for conf in zip(alg_names, n_iters, Ks, no_comprs, batch_size_for_worker, technical_batch_size, gamma, ps):
        a, n, K, nc, bw, tb, g, p = conf
        if iter_id in do_iters:
            main(algo_name=a, n_iters=n,
                 K=K, no_compr=nc, gamma=g, p=p, batch_size_for_worker=bw,
                 technical_batch_size=tb, device_num=device_num,
                 debug=DEBUG,
                 log_every=log_every)
        iter_id += 1
