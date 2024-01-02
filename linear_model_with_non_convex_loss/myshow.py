import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

import utils
import compressors


files = os.listdir('.')
logfiles = []
for file in files:
    if re.match('experiment_(.*).bin', file):
        logfiles.append(file)

KTests = len(logfiles)
Workers = 5


# ===========================================================================


main_fig_p1 = plt.figure(figsize=(12, 8))
main_fig_p2 = plt.figure(figsize=(12, 8))

oracles_fig_p1 = plt.figure(figsize=(12, 8))
oracles_fig_p2 = plt.figure(figsize=(12, 8))

main_description = ""
aux_description = ""

for t, file in enumerate(logfiles):
    run_data = utils.deserialize(file)
    fn_train_with_regul_loss = run_data['fn_train_with_regul_loss']
    fn_train_with_regul_loss_grad_norm = run_data['fn_train_with_regul_loss_grad_norm']
    descr = run_data['descr']
    KMax = run_data['KMax']
    KSamplesMax = run_data['KSamplesMax']
    KSamples = run_data['KSamples']
    transfered_bits_by_nodes = run_data['transfered_bits_by_nodes']
    fi_grad_calcs_by_nodes = run_data['fi_grad_calcs_by_nodes']

    # Unpack configuration used to launch specific test
    gamma = descr["gamma"]
    p = descr["p"]
    lamb = descr["lamb"]
    component_bits_size = descr["component_bits_size"]
    i_use_vr_marina = descr["use_vr_marina"]
    i_use_marina = descr["use_marina"]
    i_use_vr_diana = descr["use_vr_diana"]
    i_use_diana = descr["use_diana"]
    i_use_gd = descr["use_gd"]

    specified_algorihms = int(i_use_vr_marina) + int(i_use_marina) + int(i_use_vr_diana) + int(i_use_diana) + int(
        i_use_gd)

    if specified_algorihms != 1:
        print("logfile containts > 1 algorithm")
        sys.exit(-1)

    compr = compressors.Compressor()
    descr["init_compressor"](compr)
    compressor_name = compr.name()

    transfered_bits_dx = np.zeros(KMax)
    transfered_bits_mx = np.zeros(KMax)

    fi_grad_calcs_mx = np.zeros(KMax)
    fi_grad_calcs_dx = np.zeros(KMax)

    # transfered_bits_by_nodes and fi_grad_calcs_by_nodes has shape: ((Workers,KTests,KRounds,KMax))
    for z in range(KMax):
        transfered_bits_mx[z] = np.mean(transfered_bits_by_nodes[:, t, 0, z])
        transfered_bits_dx[z] = np.mean((transfered_bits_by_nodes[:, t, 0, z] - transfered_bits_mx[z]) ** 2)

    for z in range(KMax):
        fi_grad_calcs_mx[z] = np.mean(fi_grad_calcs_by_nodes[:, t, 0, z])
        fi_grad_calcs_dx[z] = np.mean((fi_grad_calcs_by_nodes[:, t, 0, z] - fi_grad_calcs_mx[z]) ** 2)

    transfered_bits_mean = np.sum(transfered_bits_by_nodes[:, t, 0, :], axis=0) / Workers
    fi_grad_calcs_sum = np.sum(fi_grad_calcs_by_nodes[:, t, 0, :], axis=0)

    for i in range(1, KMax):
        transfered_bits_mean[i] = transfered_bits_mean[i] + transfered_bits_mean[i - 1]

    for i in range(1, KMax):
        fi_grad_calcs_sum[i] = fi_grad_calcs_sum[i] + fi_grad_calcs_sum[i - 1]

    if i_use_vr_marina: prefix4algo = "vr_marina"
    if i_use_marina:    prefix4algo = "marina"
    if i_use_vr_diana:  prefix4algo = "vr_diana"
    if i_use_diana:     prefix4algo = "diana"
    if i_use_gd:        prefix4algo = "gd"

    # ===============================================================================
    # Figures
    # ===============================================================================
    algo_name = ""
    short_algo_name = ""

    if i_use_vr_marina: algo_name = f'VR-MARINA [$\\gamma$={gamma:g}, p={p:g}, batch={vr_batch_size_percentage * 100:g}% of local data]. Compress: {compressor_name}'
    if i_use_marina:    algo_name = f'MARINA [$\\gamma$={gamma:g}, p={p:g}]. Compress: {compressor_name}'
    if i_use_diana:     algo_name = f'DIANA  [$\\gamma$={gamma:g}, $\\alpha$={fixed_alpha_diana}]. Compress: {compressor_name}'
    if i_use_vr_diana:  algo_name = f'VR-DIANA [$\\gamma$={gamma:g},$\\alpha$={fixed_alpha_diana}]. Compressor: {compressor_name}'
    if i_use_gd:        algo_name = f'GD [$\\gamma$={gamma:g}]'

    if i_use_vr_marina: short_algo_name = f'VR-MARINA {compressor_name}'
    if i_use_marina:    short_algo_name = f'MARINA {compressor_name}'
    if i_use_diana:     short_algo_name = f'DIANA {compressor_name}'
    if i_use_vr_diana:  short_algo_name = f'VR-DIANA {compressor_name}'
    if i_use_gd:        short_algo_name = f'GD'

    markevery = [int(mark_mult * KMax / 10.0), int(mark_mult * KMax / 13.0), int(mark_mult * KMax / 7.0),
                 int(mark_mult * KMax / 9.0), int(mark_mult * KMax / 11.0),
                 int(mark_mult * KMax / 13.0)][t % 6]
    marker = ["x", "^", "*", "x", "^", "*", ][t % 6]
    color = ["#e41a1c", "#377eb8", "#4daf4a", "#e41a1c", "#377eb8", "#4daf4a"][t % 6]
    linestyle = ["solid", "solid", "solid", "dashed", "dashed", "dashed"][t % 6]

    # ===================================================================================================================================
    fig = plt.figure(main_fig_p1.number)
    ax = fig.add_subplot(1, 1, 1)

    print('DBG: ')
    print(KSamples)
    print(fn_train_with_regul_loss)

    ax.semilogy(KSamples, fn_train_with_regul_loss, color=color, marker=marker, markevery=markevery,
                linestyle=linestyle, label=short_algo_name)
    if t == KTests - 1:
        ax.set_xlabel('Iteration', fontdict={'fontsize': 35})
        ax.set_ylabel('$f(x)$', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    fig = plt.figure(main_fig_p2.number)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(KSamples, fn_train_with_regul_loss_grad_norm, color=color, marker=marker, markevery=markevery,
                linestyle=linestyle, label=short_algo_name)
    if t == KTests - 1:
        ax.set_xlabel('Iteration', fontdict={'fontsize': 35})
        ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    # ===================================================================================================================================
    fig = plt.figure(transport_fig_p1.number)
    ax = fig.add_subplot(1, 1, 1)
    if not i_use_gd:
        ax.semilogy(get_subset(transfered_bits_mean, KSamples), fn_train_with_regul_loss, color=color, marker=marker,
                    markevery=markevery, linestyle=linestyle, label=short_algo_name)
    if t == KTests - 1:
        ax.set_xlabel(f'#bits/n', fontdict={'fontsize': 35})
        ax.set_ylabel('$f(x)$', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    fig = plt.figure(transport_fig_p2.number)
    ax = fig.add_subplot(1, 1, 1)
    if not i_use_gd:
        ax.semilogy(get_subset(transfered_bits_mean, KSamples), fn_train_with_regul_loss_grad_norm, color=color,
                    marker=marker, markevery=markevery, linestyle=linestyle, label=short_algo_name)
    if t == KTests - 1:
        ax.set_xlabel(f'#bits/n', fontdict={'fontsize': 35})
        ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    # =====================================================================================================================================
    fig = plt.figure(oracles_fig_p1.number)
    ax = fig.add_subplot(1, 1, 1)
    epochs = (fi_grad_calcs_sum * 1.0) / (Workers * SamplePerNodeTrain)

    ax.semilogy(get_subset(epochs, KSamples), fn_train_with_regul_loss, color=color, marker=marker, markevery=markevery,
                linestyle=linestyle, label=short_algo_name)
    if t == KTests - 1:
        ax.set_xlabel(f'# epochs', fontdict={'fontsize': 35})
        ax.set_ylabel('$f(x)$', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    fig = plt.figure(oracles_fig_p2.number)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(get_subset(epochs, KSamples), fn_train_with_regul_loss_grad_norm, color=color, marker=marker,
                markevery=markevery, linestyle=linestyle, label=short_algo_name)
    if t == KTests - 1:
        main_description = main_description + algo_name
        ax.set_xlabel(f'# epochs', fontdict={'fontsize': 35})
        ax.set_ylabel('$||\\nabla f(x)||^2$', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        print("INFO: PLOT NAME:", f'"{test_name}", n={Workers}, d={D}; ' + main_description)
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    else:
        main_description = main_description + algo_name + ", "
    # ax.semilogy([0, 1], [0.25, 0.25], color='black', lw=2, transform = plt.gcf().transFigure, clip_on = False)
    # =====================================================================================================================================
    fig = plt.figure(aux_fig_p1.number)
    ax = fig.add_subplot(1, 1, 1)
    # ax.fill_between(range(1,KMax), fi_grad_calcs_mx[1:] - 3*(fi_grad_calcs_dx[1:]**0.5), fi_grad_calcs_mx[1:] + 3*(fi_grad_calcs_dx[1:]**0.5), color='#539ecd')
    ax.plot(range(0, KMax - 1), fi_grad_calcs_mx[:-1], color=color, marker=marker, markevery=markevery,
            linestyle=linestyle, label=short_algo_name)

    # 'Oracle gradient calculation per iteration
    if t == KTests - 1:
        ax.set_xlabel('Iteration', fontdict={'fontsize': 35})
        ax.set_ylabel('Oracle request for evaluate $\\nabla f_i(x)$ at iteration', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        #            fig.suptitle(f'{test_name}')
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

    fig = plt.figure(aux_fig_p2.number)
    ax = fig.add_subplot(1, 1, 1)

    # ax.fill_between(range(1,KMax), transfered_bits_mx[1:] - 3*(transfered_bits_dx[1:]**0.5), transfered_bits_mx[1:] + 3*(transfered_bits_dx[1:]**0.5), color='#539ecd')
    ax.plot(range(0, KMax - 1), transfered_bits_mx[:-1], color=color, marker=marker, markevery=markevery,
            linestyle=linestyle, label=short_algo_name)

    # Send bits per iteration
    if t == KTests - 1:
        ax.set_xlabel('Iteration', fontdict={'fontsize': 35})
        ax.set_ylabel('Sent bits from node to master at iteration', fontdict={'fontsize': 35})
        ax.grid(True)
        ax.legend(loc='best', fontsize=25)
        #           fig.suptitle(f'{test_name}')
        plt.title(f'{test_name}', fontdict={'fontsize': 35})
        plt.xticks(fontsize=27)
        plt.yticks(fontsize=30)

main_fig_p1.tight_layout()
main_fig_p2.tight_layout()
oracles_fig_p1.tight_layout()
oracles_fig_p1.tight_layout()


save_to = script_name + "_" + os.path.basename(test_name) + "_main" + "_p1" + ".pdf"
main_fig_p1.savefig(save_to, bbox_inches='tight')
print("Image is saved into: ", save_to)

save_to = script_name + "_" + os.path.basename(test_name) + "_main" + "_p2" + ".pdf"
main_fig_p2.savefig(save_to, bbox_inches='tight')
print("Image is saved into: ", save_to)

save_to = script_name + "_" + os.path.basename(test_name) + "_oracles" "_p1" + ".pdf"
oracles_fig_p1.savefig(save_to, bbox_inches='tight')
print("Image is saved into: ", save_to)

save_to = script_name + "_" + os.path.basename(test_name) + "_oracles" "_p2" + ".pdf"
oracles_fig_p2.savefig(save_to, bbox_inches='tight')
print("Image is saved into: ", save_to)




