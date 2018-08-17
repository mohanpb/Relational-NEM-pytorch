#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # no INFO/WARN logs from Tensorflow

import time
import threading
import numpy as np
import torch
from torch.utils.data import DataLoader
#from torch.distributions.bernoulli import Bernoulli

import utils
from sacred import Experiment
from sacred.utils import get_by_dotted_path
from datasets import ds
from datasets import InputDataset, collate
from nem_model import nem, NEMCell, static_nem_iterations, get_loss_step_weights
from network import net, R_NEM

ex = Experiment("R-NEM", ingredients=[ds, nem, net])


# noinspection PyUnusedLocal
@ex.config
def cfg():
    noise = {
        'noise_type': 'bitflip',                        # noise type
        'prob': 0.2,                                    # probability of annihilating the pixel
    }
    training = {
        'optimizer': 'adam',                            # {adam, sgd, momentum, adadelta, adagrad, rmsprop}
        'params': {
            'lr': 0.001,                     # float
        },
        'max_patience': 10,                             # number of epochs to wait before early stopping
        'batch_size': 3,
        'num_workers' : 1,                              # number of data reading threads
        'max_epoch': 500,
        'clip_gradients': None,                         # maximum norm of gradients
        'debug_samples': [3, 37, 54],                   # sample ids to generate plots for (None, int, list)
        'save_epochs': [1, 5, 10, 20, 50, 100]          # at what epochs to save the model independent of valid loss
    }
    validation = {
        'batch_size': 3,
        'debug_samples': [0, 1, 2]                      # sample ids to generate plots for (None, int, list)
    }

    feed_actions = False                                # whether to feed the actions (RL) via the recurrent state
    record_grouping_score = True                        # whether to use grouping to compute ARI/AMI scores
    record_relational_loss = 'collisions'               # use {events, collisions} to compute rel. losses or None

    dt = 10                                             # how many steps to include in the last loss
    log_dir = 'debug_out'                               # directory to dump logs and debug plots
    net_path = None                                     # path of to network file to initialize weights with

    # config to control run_from_file
    run_config = {
        'usage': 'test',                                # what dataset to use {training, validation, test}
        'batch_size': 100,
        'rollout_steps': 10,
        'debug_samples': [0, 1, 2],                      # sample ids to generate plots for (None, int, list)
    }


ex.add_named_config('no_score', {'record_grouping_score': False})
ex.add_named_config('no_collisions', {'record_relational_loss': None})


@ex.capture
def add_noise(data, noise):
    noise_type = noise['noise_type']
    if noise_type in ['None', 'none', None]:
        return data

    n = torch.bernoulli(noise['prob']*torch.ones(data.size()))
    p = 2*torch.mul(data,n)
    corrupted = data + n - p  # hacky way of implementing (data XOR n)
    return corrupted


@ex.capture(prefix='training')
def set_up_optimizer(parameters, optimizer, params):
    opt = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'rmsprop': torch.optim.RMSprop
    }[optimizer](parameters, **params)
    return opt
    # if clip_gradients is not None:
        # grads_and_vars = [(tf.clip_by_norm(grad, clip_gradients), var)
        #                   for grad, var in grads_and_vars]
        # for param in model_main.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # for param in model_action_main.parameters():
        #     param.grad.data.clamp_(-1, 1)



@ex.capture
def create_curve_plots(name, plot_dict, coarse_range, fine_range, log_dir):
    import matplotlib.pyplot as plt
    fig = utils.curve_plot(plot_dict, coarse_range, fine_range)
    fig.suptitle(name)
    fig.savefig(os.path.join(log_dir, name + '_curve.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


@ex.capture
def create_debug_plots(name, debug_out, sample_indices, log_dir, debug_groups=None):
    import matplotlib.pyplot as plt

    if debug_groups is not None:
        scores, confidencess = utils.evaluate_groups_seq(debug_groups[1:], debug_out['gammas'][1:], get_loss_step_weights())
    else:
        scores, confidencess = len(sample_indices) * [0.0], len(sample_indices) * [0.0]

    # produce overview plot
    for i, nr in enumerate(sample_indices):
        fig = utils.overview_plot(i, **debug_out)
        fig.suptitle(name + ', sample {},  AMI Score: {:.3f} ({:.3f}) '.format(nr, scores[i], confidencess[i]))
        fig.savefig(os.path.join(log_dir, name + '_{}.png'.format(nr)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def populate_debug_out(session, debug_graph, pipe_line, debug_samples, name):
    idxs = debug_samples if isinstance(debug_samples, list) else [debug_samples]

    out_list = ['features']
    out_list.append('groups') if debug_graph.get('groups', None) is not None else None
    out_list.append('actions') if debug_graph.get('actions', None) is not None else None

    debug_data = pipe_line.get_debug_samples(idxs, out_list=out_list)

    feed_dict = {debug_graph['inputs']: debug_data['features']}

    if debug_data.get('groups', None) is not None:
        feed_dict[debug_graph['groups']] = debug_data['groups']

    if debug_data.get('actions', None) is not None:
        feed_dict[debug_graph['actions']] = debug_data['actions']

    debug_out = session.run(debug_graph, feed_dict=feed_dict)
    create_debug_plots(name, debug_out, idxs, debug_groups=debug_data.get('groups', None))


def run_epoch(nem_cell, optimizer, data_loader, train=True):

    losses, ub_losses, r_losses, r_ub_losses, others, others_ub, r_others, r_others_ub, ari_scores = [], [], [], [], [], [], [], [], []
    # run through the epoch
    for progress, data in enumerate(data_loader):
        # run batch
        if torch.cuda.is_available():
            features = data[0][0].cuda()
            groups = data[0][1].cuda()
            collisions= data[0][2].cuda()
        else:
            features = data[0][0]
            groups = data[0][1]
            collisions= data[0][2]

        features_corrupted = add_noise(features)

        t1 = time.time()
        out = static_nem_iterations(nem_cell, features_corrupted, features, optimizer, train, groups, collisions=collisions, actions=None)
        t2 = time.time() - t1
        print(progress, t2)
        # print("Finished static nem iteration")
        # total losses (and upperbound)
        losses.append(out[0].data.cpu().numpy())
        ub_losses.append(out[1].data.cpu().numpy())

        # total relational losses (and upperbound)
        r_losses.append(out[2].data.cpu().numpy())
        r_ub_losses.append(out[3].data.cpu().numpy())

        # other losses (and upperbound)
        # others.append(out[4].data.cpu().numpy())
        # others_ub.append(out[5].data.cpu().numpy())

        # other relational losses (and upperbound)
        # r_others.append(out[6].data.cpu().numpy())
        # r_others_ub.append(out[7].data.cpu().numpy())

        # ARI
        ari_scores.append(out[4].data.cpu().numpy())
        # ari_scores.append((0., 0., 0., 0.))

    # build log dict
    log_dict = {
        'loss': float(np.mean(losses)),
        'ub_loss': float(np.mean(ub_losses)),
        'r_loss': float(np.mean(r_losses)),
        'r_ub_loss': float(np.mean(r_ub_losses)),
        # 'others': np.mean(others, axis=0),
        # 'others_ub': np.mean(others_ub, axis=0),
        # 'r_others': np.mean(r_others, axis=0),
        # 'r_others_ub': np.mean(r_others_ub, axis=0),
        'score': np.mean(ari_scores, axis=0)
        }

    return log_dict

def run_val_epoch(nem_cell, optimizer, data_loader):

    losses, ub_losses, r_losses, r_ub_losses, others, others_ub, r_others, r_others_ub, ari_scores = [], [], [], [], [], [], [], [], []
    # run through the epoch
    with torch.no_grad():
        for progress, data in enumerate(data_loader):
            # run batch
            if torch.cuda.is_available():
                features = data[0][0].cuda()
                groups = data[0][1].cuda()
                collisions= data[0][2].cuda()
            else:
                features = data[0][0]
                groups = data[0][1]
                collisions= data[0][2]

            features_corrupted = add_noise(features)

            t1 = time.time()
            out = static_nem_iterations(nem_cell, features_corrupted, features, optimizer, False, groups, collisions=collisions, actions=None)
            t2 = time.time() - t1
            print(progress, t2)
            # print("Finished static nem iteration")
            # total losses (and upperbound)
            losses.append(out[0].data.cpu().numpy())
            ub_losses.append(out[1].data.cpu().numpy())

            # total relational losses (and upperbound)
            r_losses.append(out[2].data.cpu().numpy())
            r_ub_losses.append(out[3].data.cpu().numpy())

            # other losses (and upperbound)
            # others.append(out[4].data.cpu().numpy())
            # others_ub.append(out[5].data.cpu().numpy())

            # other relational losses (and upperbound)
            # r_others.append(out[6].data.cpu().numpy())
            # r_others_ub.append(out[7].data.cpu().numpy())

            # ARI
            ari_scores.append(out[4].data.cpu().numpy())
            # ari_scores.append((0., 0., 0., 0.))

    # build log dict
    log_dict = {
        'loss': float(np.mean(losses)),
        'ub_loss': float(np.mean(ub_losses)),
        'r_loss': float(np.mean(r_losses)),
        'r_ub_loss': float(np.mean(r_ub_losses)),
        # 'others': np.mean(others, axis=0),
        # 'others_ub': np.mean(others_ub, axis=0),
        # 'r_others': np.mean(r_others, axis=0),
        # 'r_others_ub': np.mean(r_others_ub, axis=0),
        'score': np.mean(ari_scores, axis=0)
        }

    return log_dict


@ex.capture
def add_log(key, value, _run):
    if 'logs' not in _run.info:
        _run.info['logs'] = {}
    logs = _run.info['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


@ex.capture
def get_logs(key, _run):
    logs = _run.info.get('logs', {})
    return get_by_dotted_path(logs, key)


def log_log_dict(usage, log_dict):
    for log_key, value in log_dict.items():
        add_log('{}.{}'.format(usage, log_key), value)


def print_log_dict(log_dict, usage, t, dt, s_loss_weights, dt_s_loss_weights):
    print("%s Loss: %.3f (UB: %.3f), Relational Loss: %.3f (UB: %.3f), Score: %.3f took %.3fs" % (usage, log_dict['loss'], log_dict['ub_loss'], log_dict['r_loss'], log_dict['r_ub_loss'], log_dict['score'], time.time() - t))

    # print("    other losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
    #      (log_dict['others'][:, i].sum(0) / s_loss_weights, log_dict['others_ub'][:, i].sum(0) / s_loss_weights)
    #       for i in range(len(log_dict['others'][0]))])))
    # print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
    #      (log_dict['others'][-dt:, i].sum(0) / dt_s_loss_weights,
    #       log_dict['others_ub'][-dt:, i].sum(0) / dt_s_loss_weights) for i in range(len(log_dict['others'][0]))])))

    #print("    other relational losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
    #      (log_dict['r_others'][:, i].sum(0) / s_loss_weights, log_dict['r_others_ub'][:, i].sum(0) / s_loss_weights)
    #       for i in range(len(log_dict['r_others'][0]))])))
    #print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
    #      (log_dict['r_others'][-dt:, i].sum(0) / dt_s_loss_weights,
    #       log_dict['r_others_ub'][-dt:, i].sum(0) / dt_s_loss_weights) for i in range(len(log_dict['r_others'][0]))])))


@ex.automain
def run(record_grouping_score, record_relational_loss, feed_actions, net_path, training, validation, nem, dt, seed, log_dir, _run):
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    save_epochs = training['save_epochs']

    # clear debug dir
    if log_dir and net_path is None:
        utils.create_directory(log_dir)
        utils.delete_files(log_dir, recursive=True)

    # prep weights for print out
    loss_step_weights = get_loss_step_weights()
    s_loss_weights = np.sum(loss_step_weights)
    dt_s_loss_weights = np.sum(loss_step_weights[-dt:])

    # Set up data pipelines
    nr_iters = nem['nr_steps'] + 1
    out_list = ['features']
    out_list.append('groups') if record_grouping_score else None
    out_list.append(record_relational_loss) if record_relational_loss else None
    out_list.append('actions') if feed_actions else None

    train_dataset = InputDataset("training", training['batch_size'], out_list, sequence_length = nem['nr_steps'] + 1)
    valid_dataset = InputDataset("validation", validation['batch_size'], out_list, sequence_length = nem['nr_steps'] + 1)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=1,
                        shuffle=False, num_workers=training['num_workers'],
                        collate_fn=collate)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=1,
                        shuffle=False, num_workers=training['num_workers'],
                        collate_fn=collate)
    
    # Get dimensions
    input_shape = train_dataset._data_in_file['features'].shape
    W, H, C = list(input_shape)[-3:]

    inner_cell = R_NEM(nem['k'])
    nem_cell = NEMCell(inner_cell, input_shape=(W, H, C), distribution=nem['pixel_dist'])
    optimizer = set_up_optimizer(list(nem_cell.parameters())+list(inner_cell.parameters()))

    best_valid_loss = np.inf
    best_valid_epoch = 0
    for epoch in range(1, training['max_epoch'] + 1):
        # run train epoch
        t = time.time()
        log_dict = run_epoch(nem_cell, optimizer, train_data_loader, train=True)

        # log all items in dict
        log_log_dict('training', log_dict)

        # produce print-out
        print("\n" + 80 * "%" + "    EPOCH {}   ".format(epoch) + 80 * "%")
        print_log_dict(log_dict, 'Train', t, dt, s_loss_weights, dt_s_loss_weights)

        # run valid epoch
        t = time.time()
        log_dict = run_val_epoch(nem_cell, optimizer, valid_data_loader)

        # add logs
        log_log_dict('validation', log_dict)

        # produce plots
        create_curve_plots('loss', {'training': get_logs('training.loss'),
            'validation': get_logs('validation.loss')}, [0, 1000], [0, 200])
        create_curve_plots('r_loss', {'training': get_logs('training.r_loss'),
                                      'validation': get_logs('validation.r_loss')}, [0, 100], [0, 20])

        create_curve_plots('score', {'training': get_logs('training.score'),
                                     'validation': get_logs('validation.score')}, [0, 1], None)

        # produce print-out
        print("\n")
        print_log_dict(log_dict, 'Validation', t, dt, s_loss_weights, dt_s_loss_weights)

        if log_dict['loss'] < best_valid_loss:
            best_valid_loss = log_dict['loss']
            best_valid_epoch = epoch
            _run.result = float(log_dict['score']), \
                          float(log_dict['loss']), float(log_dict['ub_loss']),  float(log_dict['r_loss']), float(log_dict['r_ub_loss'])

                          #float(np.sum(log_dict['others'][-dt:, 1])/dt_s_loss_weights), \
                          #float(np.sum(log_dict['others_ub'][-dt:, 1]) / dt_s_loss_weights), \
                          #float(np.sum(log_dict['others'][-dt:, 2]) / dt_s_loss_weights), \
                          #float(np.sum(log_dict['others_ub'][-dt:, 2]) / dt_s_loss_weights), \
                          #float(np.sum(log_dict['r_others'][-dt:, 1]) / dt_s_loss_weights), \
                          #float(np.sum(log_dict['r_others_ub'][-dt:, 1]) / dt_s_loss_weights), \
                          #float(np.sum(log_dict['r_others'][-dt:, 2]) / dt_s_loss_weights), \
                          #float(np.sum(log_dict['r_others_ub'][-dt:, 2]) / dt_s_loss_weights)

            print("    Best validation loss improved to %.03f" % best_valid_loss)
            torch.save(nem_cell.state_dict(), os.path.abspath(os.path.join(log_dir, 'best')))
            print("    Saved to:", os.path.abspath(os.path.join(log_dir, 'best')))
        if epoch in save_epochs:
            torch.save(nem_cell.state_dict(), os.path.abspath(os.path.join(log_dir, 'epoch_{}'.format(epoch))))
            print("    Saved to:", os.path.abspath(os.path.join(log_dir, 'epoch_{}'.format(epoch))))

        best_valid_loss = min(best_valid_loss, log_dict['loss'])

        if best_valid_loss < np.min(get_logs('validation.loss')[-training['max_patience']:]):
            print('Early Stopping because validation loss did not improve for {} epochs'.format(training['max_patience']))
            break

        if np.isnan(log_dict['loss']):
            print('Early Stopping because validation loss is nan')
            break

    # gather best results
    best_valid_score = float(get_logs('validation.score')[best_valid_epoch - 1])
    #best_valid_score_last = float(get_logs('validation.score_last')[best_valid_epoch - 1])

    best_valid_loss = float(get_logs('validation.loss')[best_valid_epoch - 1])
    best_valid_ub_loss = float(get_logs('validation.ub_loss')[best_valid_epoch - 1])

    #best_valid_intra_loss = float(np.sum(get_logs('validation.others')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)
    #best_valid_intra_ub_loss = float(np.sum(get_logs('validation.others_ub')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)

    #best_valid_inter_loss = float(np.sum(get_logs('validation.others')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)
    #best_valid_inter_ub_loss = float(np.sum(get_logs('validation.others_ub')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)

    best_valid_r_loss = float(get_logs('validation.r_loss')[best_valid_epoch - 1])
    best_valid_r_ub_loss = float(get_logs('validation.r_ub_loss')[best_valid_epoch - 1])

    #best_valid_r_intra_loss = float(np.sum(get_logs('validation.r_others')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)
    #best_valid_r_intra_ub_loss = float(np.sum(get_logs('validation.r_others_ub')[best_valid_epoch - 1][-dt:, 1])/dt_s_loss_weights)

    #best_valid_r_inter_loss = float(np.sum(get_logs('validation.r_others')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)
    #best_valid_r_inter_ub_loss = float(np.sum(get_logs('validation.r_others_ub')[best_valid_epoch - 1][-dt:, 2])/dt_s_loss_weights)

    return best_valid_score, best_valid_loss, best_valid_ub_loss, \
           best_valid_r_loss, best_valid_r_ub_loss
