#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
import utils
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 32

import utils


class NNConfiguration: pass

def get_subset(ctr, indicies):
    return [ctr[ind] for ind in indicies]

def descr_to_dict(descr):
    lst = descr.strip().split('\n')
    res = {line.split('=')[0]: line.split('=')[1].strip() for line in lst[1:]}
    res['algo_name'] = lst[0]
    return res

def get_label(nc, descr):
    d = descr_to_dict(descr)
    algo_names = ['DIANA', 'VR-DIANA', 'MARINA', 'VR-MARINA']
    algo_name = algo_names[int(d['algo_name'])]
    N = utils.shortify(nc.KMax)
    gamma = nc.gamma
    D = nn_config.D
    K = int(d['K']) if d['K'] != 'None' else D
    p = nn_config.p

    return (f'{algo_name}: N={N}, K={utils.shortify(K)}, K={K/D:.4f}D, gamma={gamma},'
            f'p={p:.4f}')

#===================================================================================================


n_last = 9
all_files = utils.get_all_log_files()
files = utils.get_all_log_files(ids=list(range(len(all_files) - n_last, len(all_files))))

fsize = (45, 20)
main_fig   = plt.figure(figsize=fsize)
grad_fig   = plt.figure(figsize=fsize)
bits_fig_1 = plt.figure(figsize=fsize)
bits_fig_2 = plt.figure(figsize=fsize)
main_fig_epochs   = plt.figure(figsize=fsize)
grad_fig_epochs  = plt.figure(figsize=fsize)

loc = 'upper right'

axs = dict()

def get_fig_ax(number):
    fig = plt.figure(number)
    # get an existing ax for that fig, or create if it doesnt exist yet
    if number not in axs:
        axs[number] = fig.add_subplot(1, 1, 1)
    return fig, axs[number]

colors = ["#dc143c", "#00008b", "#006400", "#dc143c", "#00008b", "#006400", "#c51b7d", "#000000", "#ff5e13",
          "#dc143c", "#00008b", "#006400", "#dc143c", "#00008b", "#006400", "#c51b7d", "#000000", "#ff5e13"]
# linestyle = ["solid", "solid", "solid", "dashed", "dashed", "dashed", "dotted", "dotted", "dotted"]

for g, (dfile, lfile) in enumerate(files):
    my = utils.deserialize(lfile)
    descr = open(dfile, 'r').read()

    transfered_bits_by_node = my["transfered_bits_by_node"]
    fi_grad_calcs_by_node   = my["fi_grad_calcs_by_node"]
    train_loss              = my["train_loss"]
    test_loss               = my["test_loss"]
    fn_train_loss_grad_norm = my["fn_train_loss_grad_norm"]
    fn_test_loss_grad_norm  = my["fn_test_loss_grad_norm"]
    nn_config               = my["nn_config"]
    current_data_and_time   = my["current_data_and_time"]
    experiment_description  = my["experiment_description"]
    compressor              = my["compressors"]
    nn_config               = my["nn_config"]
    compressors_rand_K      = my["compressors_rand_K"]
    label               = get_label(nn_config, descr)


    color = colors[g]
    linestyle = 'solid' if 'MARINA' in label else 'dotted'


    freq = 1
    train_loss = [train_loss[i] for i in range(len(train_loss)) if i % freq == 0]
    test_loss  = [test_loss[i]  for i in range(len(test_loss))  if i % freq == 0]
    fn_train_loss_grad_norm  = [fn_train_loss_grad_norm[i]  for i in range(len(fn_train_loss_grad_norm))  if i % freq == 0]
    fn_test_loss_grad_norm   = [fn_test_loss_grad_norm[i]   for i in range(len(fn_test_loss_grad_norm))   if i % freq == 0]

    #===================================================================================================
    print("==========================================================")
    print(f"Informaion about experiment results '{lfile, dfile}'")
    print(f"  Content has been created at '{current_data_and_time}'")
    print(f"  Experiment description: {experiment_description}")
    print(f"  Dimension of the optimization proble: {nn_config.D}")
    print(f"  Compressor RAND-K K: {compressors_rand_K}")
    print(f"  Number of Workers: {nn_config.kWorkers}")
    print(f"  Used step-size: {nn_config.gamma}")
    print()
    print("Whole config")
    for k in dir(nn_config):
        v = getattr(nn_config, k)
        if type(v) == int or type(v) == float:
            print(" ", k, "=", v)
    print("==========================================================")

    #=========================================================================================================================
    KMax = nn_config.KMax

    fi_grad_calcs_sum      = np.sum(fi_grad_calcs_by_node, axis = 0)
    transfered_bits_sum    = np.sum(transfered_bits_by_node, axis = 0)

    for i in range(1, KMax):
        transfered_bits_sum[i] = transfered_bits_sum[i] + transfered_bits_sum[i-1]

    transfered_bits_mean = transfered_bits_sum / nn_config.kWorkers

    for i in range(1, KMax):
        fi_grad_calcs_sum[i] = fi_grad_calcs_sum[i] + fi_grad_calcs_sum[i-1]

    transfered_bits_mean_sampled = [transfered_bits_mean[i] for i in range(len(transfered_bits_mean)) if i % freq == 0]

    #=========================================================================================================================

    epochs = (fi_grad_calcs_sum * 1.0) / (nn_config.train_set_full_samples)
    iterations = range(KMax)

    iterations_sampled =  [iterations[i] for i in range(len(iterations)) if i % freq == 0]
    epochs_sampled     =  [epochs[i] for i in range(len(epochs)) if i % freq == 0]

    #=========================================================================================================================

    fig, ax = get_fig_ax(main_fig.number)
    ax.semilogy(iterations_sampled, train_loss, color=color,
                                                linestyle=linestyle, label=label)


    ax.set_xlabel('Communication Rounds', fontdict = {'fontsize':35})
    ax.set_ylabel('$f(x)$', fontdict = {'fontsize':35})

    ax.grid(True)
    ax.legend(loc=loc, fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # fig.tight_layout()
    #=========================================================================================================================
    fig, ax = get_fig_ax(grad_fig.number)

    ax.semilogy(iterations_sampled, fn_train_loss_grad_norm, color=color, linestyle=linestyle, label=label)
    ax.set_xlabel('Communication Rounds', fontdict = {'fontsize':35})
    ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})

    ax.grid(True)
    ax.legend(loc=loc, fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # fig.tight_layout()
    #=========================================================================================================================
    fig, ax = get_fig_ax(bits_fig_1.number)

    ax.semilogy(transfered_bits_mean_sampled, train_loss, color=color, linestyle=linestyle, label=label)
    ax.set_xlabel('#bits/n', fontdict = {'fontsize':35})
    ax.set_ylabel('f(x)', fontdict = {'fontsize':35})

    ax.grid(True)
    ax.legend(loc=loc, fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # fig.tight_layout()
    #=========================================================================================================================
    fig, ax = get_fig_ax(bits_fig_2.number)

    #g = (g + 1)%len(color)
    ax.semilogy(transfered_bits_mean_sampled, fn_train_loss_grad_norm, color=color, linestyle=linestyle, label=label)

    ax.set_xlabel('#bits/n', fontdict = {'fontsize':35})
    ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})
    ax.grid(True)
    ax.legend(loc=loc, fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # fig.tight_layout()
    #=========================================================================================================================
    fig, ax = get_fig_ax(main_fig_epochs.number)
    ax.semilogy(epochs_sampled, train_loss, color=color,
                                                linestyle=linestyle, label=label)


    ax.set_xlabel('Epochs', fontdict = {'fontsize':35})
    ax.set_ylabel('$f(x)$', fontdict = {'fontsize':35})

    ax.grid(True)
    ax.legend(loc=loc, fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # fig.tight_layout()
    
    #=========================================================================================================================
    
    fig, ax = get_fig_ax(grad_fig_epochs.number)

    ax.semilogy(epochs_sampled, fn_train_loss_grad_norm, color=color, linestyle=linestyle, label=label)
    ax.set_xlabel('Epochs', fontdict = {'fontsize':35})
    ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict = {'fontsize':35})

    ax.grid(True)
    ax.legend(loc=loc, fontsize = 25)
    plt.title(f'{experiment_description}', fontdict = {'fontsize':35})
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # # fig.tight_layout()

# main_# fig.tight_layout()
# grad_# fig.tight_layout()
# bits_fig_1.tight_layout()
# bits_fig_2.tight_layout()
# main_fig_epochs.tight_layout()
# grad_fig_epochs.tight_layout()

save_to = "1_main_fig.pdf"
main_fig.savefig(save_to, bbox_inches='tight')

save_to = "2_grad_fig.pdf"
grad_fig.savefig(save_to, bbox_inches='tight')

save_to = "3_bits_fig_1.pdf"
bits_fig_1.savefig(save_to, bbox_inches='tight')

save_to = "4_bits_fig_2.pdf"
bits_fig_2.savefig(save_to, bbox_inches='tight')

save_to = "5_main_fig_epochs.pdf"
main_fig_epochs.savefig(save_to, bbox_inches='tight')

save_to = "6_grad_fig_epochs.pdf"
grad_fig_epochs.savefig(save_to, bbox_inches='tight')

