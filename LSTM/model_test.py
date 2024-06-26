# -*- coding:utf-8 -*-

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process import device, get_mape, setup_seed, MyDataset
from model_train import load_data
from models import LSTM, BiLSTM, Seq2Seq, MTL_LSTM
# import seaborn as sns



setup_seed(20)


def test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


def m_test(args, Dtes, PATHS, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    Dtes = [[x for x in iter(Dte)] for Dte in Dtes]
    models = []
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        model.load_state_dict(torch.load(path)['models'])
        model.eval()
        models.append(model)
    print('predicting...')
    for i in range(len(Dtes[0])):
        for j in range(len(Dtes)):
            model = models[j]
            seq, label = Dtes[j][i][0], Dtes[j][i][1]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            seq = seq.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    # print(y, pred)
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


def seq2seq_test(args, Dte, path, m, n):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


def list_of_groups(data, sub_len):
    groups = zip(*(iter(data),) * sub_len)
    end_list = [list(i) for i in groups]
    count = len(data) % sub_len
    end_list.append(data[-count:]) if count != 0 else end_list
    return end_list


def ss_rolling_test(args, Dte, path, m, n):
    """
    :param args:
    :param Dte:
    :param path:
    :param m:
    :param n:
    :return:
    """
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    #
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[len(sub_pred) - len(seq) + t]
                else:
                    for t in range(len(sub_pred)):
                        seq[len(seq) - len(sub_pred) + t][0] = sub_pred[t]
            else:
                seq = seq.cpu().numpy().tolist()[0]
            # print(new_seq)
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=6)  # 2080Ti的num_worker数量为6
            # print(new_seq)
            seq = [x for x in iter(seq)][0]
            # print(new_seq)
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # print(y_pred)
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)


# multiple models scrolling
def mms_rolling_test(args, Dte, PATHS, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    models = []
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        model.load_state_dict(torch.load(path)['models'])
        model.eval()
        models.append(model)

    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    #
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            model = models[seq_idx]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[len(sub_pred) - len(seq) + t]
                else:
                    for t in range(len(sub_pred)):
                        seq[len(seq) - len(sub_pred) + t][0] = sub_pred[t]
            else:
                seq = seq.cpu().numpy().tolist()[0]
            # print(new_seq)
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=6)  # 2080Ti的num_worker数量为6
            # print(new_seq)
            seq = [x for x in iter(seq)][0]
            # print(new_seq)
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # print(y_pred)
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


# def rolling_test(args, path, flag):
#     """
#     SingleStep Rolling Predicting.
#     :param args:
#     :param path:
#     :param flag:
#     :return:
#     """
#     Dtr, Dte, m, n = load_data(args, flag, args.batch_size)
#     pred = []
#     y = []
#     print('loading models...')
#     input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
#     output_size = args.output_size
#     if args.bidirectional:
#         model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
#     else:
#         model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
#     # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
#     model.load_state_dict(torch.load(path)['models'])
#     model.eval()
#     print('predicting...')
#     for batch_idx, (seq, target) in enumerate(tqdm(Dte), 0):
#         target = list(chain.from_iterable(target.data.tolist()))
#         y.extend(target)
#         if batch_idx != 0:
#             seq = seq.cpu().numpy().tolist()[0]
#             if len(pred) >= len(seq):
#                 x = [[x] for x in pred]
#                 new_seq = x[-len(seq):]
#             else:
#                 new_seq = seq[:(len(seq) - len(pred))]
#                 x = [[x] for x in pred]
#                 new_seq.extend(x)
#         else:
#             new_seq = seq.cpu().numpy().tolist()[0]
#         # print(new_seq)
#         new_seq = [new_seq]
#         new_seq = torch.FloatTensor(new_seq)
#         new_seq = MyDataset(new_seq)
#         new_seq = DataLoader(dataset=new_seq, batch_size=1, shuffle=False, num_workers=0)
#         # print(new_seq)
#         new_seq = [x for x in iter(new_seq)][0]
#         # print(new_seq)
#         with torch.no_grad():
#             new_seq = new_seq.to(device)
#             y_pred = model(new_seq)
#             y_pred = list(chain.from_iterable(y_pred.data.tolist()))
#             # print(y_pred)
#             pred.extend(y_pred)
#
#     y, pred = np.array(y), np.array(pred)
#     y = (m - n) * y + n
#     pred = (m - n) * pred + n
#     print('mape:', get_mape(y, pred))
#     plot(y, pred)


def mtl_test(args, Dte, scaler, path):
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = MTL_LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size,
                     n_outputs=args.n_outputs).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.n_outputs)]
    preds = [[] for i in range(args.n_outputs)]
    for (seq, targets) in tqdm(Dte):
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(args.n_outputs):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            _pred = model(seq)
            for i in range(_pred.shape[0]):
                pred = _pred[i]
                pred = list(chain.from_iterable(pred.data.tolist()))
                preds[i].extend(pred)

    # ys, preds = [np.array(y) for y in ys], [np.array(pred) for pred in preds]
    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('mape_loss = :', get_mape(y, pred))
        # mtl_plot(y, pred, ind + 1)    # 折线拟合图，可暂时不画
    # plt.show()                        # 折线拟合图，可暂时不画

    # print('target:', ys[0, :])
    # print('pred:', preds[0, :])
    # [30.973762   15.99939929 15.99939929 ...  1.00794     1.00794 12.0169996 ]
    # [31.6900497  16.37271965 16.23452713 ...  1.28853531 -0.47014774 9.96304908]
    a = ys[0, :]  # 取矩阵第一行，创建新列表
    b = preds[0, :]
    np.savetxt(rootPath + '/pycharm_project_3/algorithms/elements.csv', (a, b), delimiter=',')
    # pearsonr(a, b, 243)


def plot(y, pred):
    # plot
    x = [i for i in range(1, 350 + 1)]  # range(1,151)
    # print(len(y))
    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y[150:500])(x_smooth)  # y[150:300],数量对应上面的x_range
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:500])(x_smooth)  # y[150:300],数量对应上面的x_range
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def mtl_plot(y, pred, ind):
    # plot
    x = [i for i in range(1, 243 + 1)]  # range(1,151)
    print(len(y))
    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y[:243])(x_smooth)  # y[150:300],数量对应上面的x_range
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true' + str(ind))

    y_smooth = make_interp_spline(x, pred[:243])(x_smooth)  # y[150:300],数量对应上面的x_range
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred' + str(ind))
    plt.grid(axis='y')
    plt.legend(loc='upper center', ncol=6)


def pearsonr(y, pred, n):
    a = []  #创建列表
    b = []
    for i in range(n):  # 取前面n个
        a.append(y[i])
        b.append(pred[i])

    print(a)
    print(b)

    outputs = np.zeros(shape=(n, n))  # 创建矩阵
    def pearson(true, pred, n):
        for i in range(n):
            for j in range(n):
                outputs[i][j] = 1 - np.abs(true[i] - pred[j]) / np.abs((true[i] + pred[j]))
        print(outputs)
        np.savetxt('new.csv', outputs, delimiter=',')
        # df = pd.DataFrame(output_size)
        # df.to_csv('PearsonrMatrix.csv', index=False, header=False)
        return outputs

    # pearson(a, b, 243)
    pearson(a, b, n)
    return
