from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
import threading
from debug_utils import *
from param_utils import *


class WorkerThreadDiana(threading.Thread):
    def __init__(self, wcfg, ncfg, stats, print_lock=None):
        threading.Thread.__init__(self)
        self.wcfg = wcfg
        self.ncfg = ncfg

        self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
        self.model = self.model.to(wcfg.device)  # move model to device
        wcfg.model = self.model

        utils.setupAllParamsRandomly(self.model)

        self.stats = stats
        self.print_lock = print_lock

    def run(self):
        wcfg = self.wcfg
        ncfg = self.ncfg
        stats = self.stats

        # wcfg - configuration specific for worker
        # ncfg - general configuration with task description

        dbgprint(self.print_lock, wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ",
                 len(wcfg.train_set))
        # await init_workers_permission_event.wait()

        model = self.model

        loaders = (wcfg.train_loader, wcfg.test_loader)  # train and test loaders

        # Setup unitial shifts
        # =========================================================================================
        yk = utils.getAllParams(model)
        hk = zero_params_like(yk)
        # ========================================================================================
        # Extra constants
        # one_div_trainset_all_len = torch.Tensor([1.0/len(wcfg.train_set_full)]).to(wcfg.device)
        one_div_trainset_len = torch.Tensor([1.0 / len(wcfg.train_set)]).to(wcfg.device)
        one_div_batch_prime_len = torch.Tensor([1.0 / (ncfg.batch_size_for_worker)]).to(wcfg.device)
        delta_flatten = torch.zeros(ncfg.D).to(wcfg.device)
        # =========================================================================================
        iteration = 0
        # =========================================================================================
        full_grad_w = []
        # =========================================================================================
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
                # ================================================================================================================================
                # Generate subsample with b' cardinality
                if ncfg.i_use_vr_marina:
                    indicies = torch.randperm(len(wcfg.train_set))[0:ncfg.batch_size_for_worker]
                    subset = torch.utils.data.Subset(wcfg.train_set, indicies)
                else:
                    subset = wcfg.train_set

                dbgprint(self.print_lock, self.wcfg, 'WORKER USING SUBSET ', len(subset))

                ts.append(time())

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

                ts.append(time())

                # ================================================================================================================================
                # Evaluate in current "xk" within b' batch
                prev_train_mode = torch.is_grad_enabled()
                # ================================================================================================================================
                model.train(True)
                i = 0
                for p in model.parameters():
                    p.data.flatten(0)[:] = wcfg.input_for_cmd[i].flatten(0)
                    i += 1

                ts.append(time())

                for inputs, outputs in minibatch_loader:
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs,
                                                                     reduction='sum')  # compute objective
                    loss.backward()
                    # compute the gradient (backward-pass)

                ts.append(time())

                gk_x = []
                for p in model.parameters():
                    gk_x.append(p.grad.data.detach().clone())
                    p.grad = None

                ts.append(time())

                # ================================================================================================================================
                i = 0
                for p in model.parameters():
                    p.data.flatten(0)[:] = yk[i].flatten(0)
                    i += 1

                ts.append(time())

                for inputs, outputs in minibatch_loader:
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs,
                                                                     reduction='sum')  # compute objective
                    loss.backward()  # compute the gradient (backward-pass)

                ts.append(time())

                gk_w = []
                for p in model.parameters():
                    gk_w.append(p.grad.data.detach().clone())
                    p.grad = None
                # ================================================================================================================================

                ts.append(time())

                for inputs, outputs in wcfg.train_loader:
                    inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
                    logits = model(inputs)  # forward-pass: Make a forward pass through the network
                    loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')  # compute objective
                    loss.backward()

                if len(full_grad_w) == 0:  # compute the gradient (backward-pass)
                    for p in model.parameters():
                        full_grad_w.append(p.grad.data.detach().clone())
                        p.grad = None

                ts.append(time())

                # ================================================================================================================================
                model.train(prev_train_mode)
                # ================================================================================================================================
                # dbgprint(self.print_lock, wcfg, "gkx", len(gk_x))
                # dbgprint(self.print_lock, wcfg, "gkw", len(gk_w))
                # dbgprint(self.print_lock, wcfg, "gfull", len(full_grad_w))
                # dbgprint(self.print_lock, wcfg, "hk", hk)

                gk_next = add_params(sub_params(gk_x, gk_w), full_grad_w)
                delta = sub_params(gk_next, hk)

                ts.append(time())

                # Compress delta
                # ================================================================================================================================
                delta_offset = 0
                for t in range(len(delta)):
                    offset = len(delta[t].flatten(0))
                    delta_flatten[(delta_offset):(delta_offset + offset)] = delta[t].flatten(0)
                    delta_offset += offset

                delta_flatten = wcfg.compressor.compressVector(delta_flatten)  # Compress shifted local gradient

                delta_offset = 0
                for t in range(len(delta)):
                    offset = len(delta[t].flatten(0))
                    delta[t].flatten(0)[:] = delta_flatten[(delta_offset):(delta_offset + offset)]
                    delta_offset += offset
                # ================================================================================================================================
                mk_i = delta
                hk = add_params(hk, mult_param(ncfg.fixed_alpha_diana, mk_i))
                # ================================================================================================================================
                i = 0
                for p in model.parameters():
                    wcfg.output_of_cmd.append(mk_i[i].data.detach().clone())
                    i += 1
                # ================================================================================================================================
                stats.transfered_bits_by_node[
                    wcfg.worker_id, iteration] = wcfg.compressor.last_need_to_send_advance * ncfg.component_bits_size

                if iteration == 0:
                    # need to take full grad at beginning
                    stats.fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(wcfg.train_set)
                elif wcfg.cmd == "bcast_xk_uk_1":
                    # update control imply more big gradient at next step
                    stats.fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(
                        wcfg.train_set) + 2 * ncfg.batch_size_for_worker
                else:
                    stats.fi_grad_calcs_by_node[wcfg.worker_id, iteration] = 2 * ncfg.batch_size_for_worker

                if wcfg.cmd == "bcast_xk_uk_1":
                    for i in range(len(yk)):
                        yk[i].flatten(0)[:] = wcfg.input_for_cmd[i].flatten(0)
                    full_grad_w = []

                iteration += 1
                # ================================================================================================================================

                ts.append(time())
                dbgprint(self.print_lock, self.wcfg, [ts[i + 1] - ts[i] for i in range(len(ts) - 1)])

                wcfg.cmd_output_ready.release()
            # ===========================================================================================

        # Signal that worker has finished initialization via decreasing semaphore
        # completed_workers_semaphore.acquire()
        dbgprint(self.print_lock, wcfg, f"END")