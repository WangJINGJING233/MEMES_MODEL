# /usr/bin/python
# -*- coding=utf-8 -*-

import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import powerlaw


def generate_SF_network_edge():
    # 网络参数配置
    network_size = 100000
    min_k, max_k = 4, int(network_size ** 0.5)
    gamma = 3
    # 网络度分布概率分配
    degree_prob = [0 for i in range(min_k)] + [k ** (-gamma) for k in range(min_k, max_k + 1)]
    total_prob = sum(degree_prob)
    degree_prob = [1.0 * i / total_prob for i in degree_prob]
    node_degree_list = [0 for i in range(network_size)]
    for i in range(1, len(degree_prob)):
        degree_prob[i] += degree_prob[i - 1]
    # 赌轮法确定节点的度
    for i in range(network_size):
        random_p = random.random()
        for j in range(len(degree_prob)):
            if degree_prob[j] >= random_p:
                node_degree_list[i] = j
                break
    # 保证边数是个偶数，避免剩下一条边
    if sum(node_degree_list) % 2 != 0:
        node_degree_list[0] += 1
    # 根据节点的度构建节点数目
    node_list = []
    node_id = 0
    for i in range(network_size):
        temp = [node_id for _ in range(node_degree_list[i])]
        node_list.extend(temp)
        node_id += 1
    # 将所有的节点分为两半并打乱，然后对应连边形成邻居关系
    stubs1 = node_list[:len(node_list) // 2]
    stubs2 = node_list[len(node_list) // 2:]
    shuffle(stubs1)
    shuffle(stubs2)
    node_neighbour = [set() for _ in range(network_size)]
    for i in range(len(stubs1)):
        node1, node2 = stubs1[i], stubs2[i]
        # 存在重边或者自边时则重新寻找一个节点，直到不存在重边或者自边
        flag = (node1 == node2) or (node1 in node_neighbour[node2]) or (node2 in node_neighbour[node1])
        while (flag):
            node_index = random.randint(i + 1, len(stubs2) - 1)
            node3 = stubs2[node_index]
            flag = (node1 == node3) or (node1 in node_neighbour[node3]) or (node3 in node_neighbour[node1])
            if not flag:
                stubs2[node_index] = node2
                node2 = node3
                break
        node_neighbour[node1].add(node2)
        node_neighbour[node2].add(node1)
    node_neighbour = [list(i) for i in node_neighbour]

    network_file = 'network_SF_' + str(network_size) + '_' + str(gamma) + '.txt'
    fw = open(network_file, 'w')
    fw.write(str(node_neighbour))
    fw.close()


def generate_ER_network_edge():
    # 网络参数配置
    network_size = 100000
    k = 10
    # 根据节点的度构建节点数目
    node_list = []
    node_id = 0
    for i in range(network_size):
        node_list.extend([node_id for _ in range(k)])
        node_id += 1

    # 将所有的节点分为两半并打乱，然后对应连边形成邻居关系
    stubs1 = node_list[:len(node_list) // 2]
    stubs2 = node_list[len(node_list) // 2:]
    shuffle(stubs1)
    shuffle(stubs2)
    node_neighbour = [set() for _ in range(network_size)]
    for i in range(len(stubs1)):
        node1, node2 = stubs1[i], stubs2[i]
        # 存在重边或者自边时则重新寻找一个节点，直到不存在重边或者自边
        flag = (node1 == node2) or (node1 in node_neighbour[node2]) or (node2 in node_neighbour[node1])
        while (flag):
            node_index = random.randint(i + 1, len(stubs2) - 1)
            node3 = stubs2[node_index]
            flag = (node1 == node3) or (node1 in node_neighbour[node3]) or (node3 in node_neighbour[node1])
            if not flag:
                stubs2[node_index] = node2
                node2 = node3
                break
        node_neighbour[node1].add(node2)
        node_neighbour[node2].add(node1)
    node_neighbour = [list(i) for i in node_neighbour]
    network_file = 'network_ER_' + str(network_size) + '_' + str(k) + '.txt'
    fw = open(network_file, 'w')
    fw.write(str(node_neighbour))
    fw.close()


def plot_distribution(data, mode="PDF", start=1, color='b', linewidth=1, marker='o', markersize=8):
    "绘制数据概率分布和互补概率分布"
    data = list(filter(lambda i: i >= start, data))
    end = max(data) + 1
    x = np.arange(end)
    y = np.zeros(end)
    for i in data:
        y[i] += 1
    y = 1.0 * y / len(data)
    if mode == "PDF":
        plt.scatter(x[start:], y[start:], linewidth=linewidth, color=color, marker=marker)
    elif mode == "CCDF":
        for i in range(len(y) - 2, -1, -1):
            y[i] = y[i + 1] + y[i]
        plt.plot(x[start:], y[start:], linewidth=linewidth, color=color, marker=marker, markersize=markersize)
    else:
        assert 'plot_distribution mode error'
    plt.xscale("log")
    plt.yscale("log")


if __name__ == '__main__':
    # generate_SF_network_edge()
    network_file = open('network_SF_100000_3.txt')
    node_neighbour = eval(network_file.readline())
    data = [len(i) for i in node_neighbour]
    plot_distribution(data, mode='PDF')
    plt.axis([1, 1000, 1e-5, 1])
    plt.show()
    fit = powerlaw.Fit(data, fit_method='KS', parameter_range={'alpha': [1, 4], 'sigma': [None, 0.1]})
    print(fit.alpha)
    plt.figure()
    fit.plot_pdf()
    plt.axis([1, 1000, 1e-5, 1])
    plt.show()
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print(fit.alpha, R, p)
