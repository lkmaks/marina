# class WorkerThreadMarina(threading.Thread):
#     def __init__(self, wcfg, ncfg):
#         threading.Thread.__init__(self)
#         self.wcfg = wcfg
#         self.ncfg = ncfg
#
#         self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
#         self.model = self.model.to(wcfg.device)  # move model to device
#         utils.setupAllParamsRandomly(self.model)
#         wcfg.model = self.model
#
#     def run(self):
#         wcfg = self.wcfg
#         ncfg = self.ncfg
#
#         global transfered_bits_by_node
#         global fi_grad_calcs_by_node
#         global train_loss
#         global test_loss
#         global fn_train_loss_grad_norm
#         global fn_test_loss_grad_norm
#
#         # wcfg - configuration specific for worker
#         # ncfg - general configuration with task description
#
#         dbgprint(wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ",
#                  len(wcfg.train_set))
#         # await init_workers_permission_event.wait()
#
#         model = self.model
#
#         loaders = (wcfg.train_loader, wcfg.test_loader)  # train and test loaders
#
#         # one_div_trainset_all_len = torch.Tensor([1.0/len(wcfg.train_set_full)]).to(wcfg.device)
#         one_div_trainset_len = torch.Tensor([1.0 / len(wcfg.train_set)]).to(wcfg.device)
#
#         if ncfg.i_use_vr_marina:
#             one_div_batch_prime_len = torch.Tensor([1.0 / (ncfg.batch_size_for_worker)]).to(wcfg.device)
#         else:
#             one_div_batch_prime_len = torch.Tensor([1.0 / len(wcfg.train_set)]).to(wcfg.device)
#
#         delta_flatten = torch.zeros(ncfg.D).to(wcfg.device)
#         # =========================================================================================
#         iteration = 0
#         # =========================================================================================
#
#         while True:
#             wcfg.input_cmd_ready.acquire()
#             if wcfg.cmd == "exit":
#                 wcfg.output_of_cmd = ""
#                 wcfg.cmd_output_ready.release()
#                 break
#
#             #        if wcfg.cmd == "curr_x":
#             #            wcfg.output_of_cmd = []
#             #            for p in model.parameters():
#             #                wcfg.output_of_cmd.append(p.data.detach().clone())
#             #            wcfg.cmd_output_ready.release()
#
#             # dbgprint(self.wcfg, wcfg.cmd)
#
#             if wcfg.cmd == "bcast_g_c1":
#                 wcfg.output_of_cmd = []
#                 k = 0
#                 for p in model.parameters():
#                     p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[k]
#                     k = k + 1
#
#                 prev_train_mode = torch.is_grad_enabled()
#                 model.train(True)
#
#                 for inputs, outputs in wcfg.train_loader:
#                     inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
#                     logits = model(inputs)  # forward-pass: Make a forward pass through the network
#                     loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')  # compute objective
#                     loss.backward()  # compute the gradient (backward-pass)
#
#                 for p in model.parameters():
#                     wcfg.output_of_cmd.append(p.grad.data.detach().clone())
#                     p.grad = None
#
#                 model.train(prev_train_mode)
#
#                 # Case when we send to master really complete gradient
#                 transfered_bits_by_node[wcfg.worker_id, iteration] = ncfg.D * ncfg.component_bits_size
#                 # On that "c1" mode to evaluate gradient we call oracle number of times how much data is in local "full" batch
#                 fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(wcfg.train_set)
#
#                 wcfg.cmd_output_ready.release()
#                 iteration += 1
#
#             if wcfg.cmd == "bcast_g_c0":
#                 wcfg.output_of_cmd = []
#
#                 # ================================================================================================================================
#                 # Generate subsample with b' cardinality
#                 indicies = None
#                 if ncfg.i_use_vr_marina:
#                     indicies = torch.randperm(len(wcfg.train_set))[0:ncfg.batch_size_for_worker]
#                     subset = torch.utils.data.Subset(wcfg.train_set, indicies)
#                 else:
#                     subset = wcfg.train_set
#                 # ================================================================================================================================
#                 minibatch_loader = DataLoader(
#                     subset,  # dataset from which to load the data.
#                     batch_size=ncfg.batch_size,  # How many samples per batch to load (default: 1).
#                     shuffle=False,  # Set to True to have the data reshuffled at every epoch (default: False)
#                     drop_last=False,
#                     # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
#                     pin_memory=False,
#                     # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
#                     collate_fn=None,  # Merges a list of samples to form a mini-batch of Tensor(s)
#                 )
#
#                 # Evaluate in previous point SGD within b' batch
#                 prev_train_mode = torch.is_grad_enabled()
#                 model.train(True)
#
#                 # ================================================================================================================================
#                 from time import time
#                 t0 = time()
#                 for t, (inputs, outputs) in enumerate(minibatch_loader):
#                     t1 = time()
#                     inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
#                     logits = model(inputs)  # forward-pass: Make a forward pass through the network
#                     loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs,
#                                                                      reduction='sum')  # compute objective
#                     loss.backward()  # compute the gradient (backward-pass)
#                     t5 = time()
#                     # dbgprint(self.wcfg, f'For MB cat: {t1 - t0:.3f}; For rest: {t5 - t1:.3f}')
#
#                 g_batch_prev = []
#                 for p in model.parameters():
#                     g_batch_prev.append(p.grad.data.detach().clone())
#                     p.grad = None
#                 # ================================================================================================================================
#                 # Change xk: move from x(k) to x(k+1)
#                 k = 0
#                 for p in model.parameters():
#                     p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[k]
#                     k = k + 1
#
#                 # Evaluate SGD in the new point within b' batch
#                 # ================================================================================================================================
#                 t0 = time()
#                 for inputs, outputs in minibatch_loader:
#                     t1 = time()
#                     inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
#                     logits = model(inputs)  # forward-pass: Make a forward pass through the network
#                     loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs,
#                                                                      reduction='sum')  # compute objective
#                     loss.backward()  # compute the gradient (backward-pass)
#                     t5 = time()
#                     # dbgprint(self.wcfg, f'For MB cat: {t1 - t0:.3f}; For rest: {t5 - t1:.3f}')
#
#                 t0 = time()
#                 g_batch_next = []
#                 for p in model.parameters():
#                     g_batch_next.append(p.grad.data.detach().clone())
#                     p.grad = None
#                 # dbgprint(self.wcfg, f'For Grad clone -> g_batch_next: {time() - t0:.3f}')
#
#                 # t0 = time() # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
#                 delta = sub_params(g_batch_next, g_batch_prev)
#
#                 t1 = time()
#                 # dbgprint(self.wcfg, f'For sub params: {t1 - t0:.3f}')
#
#                 delta_offset = 0
#                 for t in range(len(delta)):
#                     offset = len(delta[t].flatten(0))
#                     delta_flatten[(delta_offset):(delta_offset + offset)] = delta[t].flatten(0)
#                     delta_offset += offset
#
#                 # dbgprint(self.wcfg, f'Delta offsetting: {time() - t1:.3f}')
#                 # t1 = time()
#
#                 delta_flatten = wcfg.compressor.compressVector(delta_flatten)
#                 # dbgprint(self.wcfg, f'Compressing delta_flatten: {time() - t1:.3f}')
#                 # t1 = time()
#
#                 delta_offset = 0
#                 for t in range(len(delta)):
#                     offset = len(delta[t].flatten(0))
#                     delta[t].flatten(0)[:] = delta_flatten[(delta_offset):(delta_offset + offset)]
#                     delta_offset += offset
#
#                 # dbgprint(self.wcfg, f'Delta offsetting p2: {time() - t1:.3f}')
#                 # t1 = time()
#
#                 g_new = add_params(wcfg.input_for_cmd, delta)
#                 wcfg.output_of_cmd = g_new
#                 model.train(prev_train_mode)
#
#                 transfered_bits_by_node[
#                     wcfg.worker_id, iteration] = wcfg.compressor.last_need_to_send_advance * ncfg.component_bits_size
#                 fi_grad_calcs_by_node[wcfg.worker_id, iteration] = 2 * ncfg.batch_size_for_worker
#                 iteration += 1
#
#                 wcfg.cmd_output_ready.release()
#
#                 # dbgprint(self.wcfg, f'For end of work: {time() - t1:.3f}')
#
#             if wcfg.cmd == "full_grad":
#                 wcfg.output_of_cmd = []
#                 prev_train_mode = torch.is_grad_enabled()
#                 model.train(True)
#
#                 for inputs, outputs in wcfg.train_loader:
#                     inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)  # move to device
#                     logits = model(inputs)  # forward-pass: Make a forward pass through the network
#                     loss = one_div_trainset_len * F.cross_entropy(logits, outputs)  # compute objective
#                     loss.backward()  # compute the gradient (backward-pass)
#
#                 for p in model.parameters():
#                     wcfg.output_of_cmd.append(p.grad.data.detach().clone())
#                     p.grad = None
#
#                 model.train(prev_train_mode)
#                 wcfg.cmd_output_ready.release()
#
#         # Signal that worker has finished initialization via decreasing semaphore
#         # completed_workers_semaphore.acquire()
#         dbgprint(wcfg, f"END")
