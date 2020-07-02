# /usr/bin/python
# -*- coding=utf-8 -*-
import os
import datetime
import collections
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy

plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100


def timer(function):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        print(time_start)
        res = function(*args, **kwargs)
        time_end = time.time()
        print(time_end)
        print(time_end - time_start)
        return res

    return wrapper


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

@timer
def model():
    "真假多信息竞争传播模型"
    # 导入网络拓扑结构
    filename = os.getcwd() + os.sep + "dat_network_directed_powerlaw.csv"
    # filename = os.getcwd() + os.sep + "dat_network_directed_zregular.csv"
    fr = open(filename, "r")
    N = int(fr.readline())  # 网络规模
    node_out = eval(fr.readline())  # 节点出度邻居，节点编号从1开始
    node_in = eval(fr.readline())  # 节点入度邻居，节点编号从1开始
    fr.close()

    # 信息传播模型参数设置

    true_tweet_prob, false_tweet_prob = 0.1, 0.4  # 真假信息的产生概率
    true_retweet_prob, false_retweet_prob = 0.2, 0.8  # 真假信息的产生概率
    L = 1  # 节点屏幕的长度
    T = 10
    end_time = int(T * N) + 1  # 传播总时间
    nodes_screen = [collections.deque([-1 for i in range(L)]) for i in range(N + 1)]  # 节点的屏幕，下标从1开始
    message_id = 0  # 信息的编号
    message_veracity = [0 for i in range(int(0.2 * N))]  # 信息的真假性初始为0，1为真信息，-1为假信息
    message_popularity = [0 for i in range(int(0.2 * N))]  # 信息的流行度
    # message_veracity = dict()
    # message_popularity = dict()
    tweet_node_list = np.random.randint(1, N + 1, end_time).tolist()  # 每个时间步随机抽取的节点
    record_list = [i * N for i in [0.1, 1, 10]]
    message_popularity_list = []

    # 信息传播数值模拟
    # fp = open("result.txt", 'w')
    # fp.write('[')
    for t in range(end_time):
        if t % N == 0: print(t / N)
        # 随机选择一个节点，获取其出度邻居节点
        tweet_node = tweet_node_list[t]
        follower = node_out[tweet_node - 1]
        # 若节点的屏幕为空，则创建按概率产生一条新的真假信息并传播给它的所有粉丝
        if nodes_screen[tweet_node][0] == -1:

            if random.random() <= true_tweet_prob:
                message_veracity[message_id] = 1
            else:
                message_veracity[message_id] = -1
            nodes_screen[tweet_node].pop()
            nodes_screen[tweet_node].appendleft(message_id)
            # 产生新的真或假信息之后必定转给其粉丝
            message_popularity[message_id] = 0
            for i in follower:
                nodes_screen[i].pop()
                nodes_screen[i].appendleft(message_id)
                message_popularity[message_id] += 1  # 转发一次流行度加1而不是加粉丝数
            message_id += 1
        # 若节点的屏幕非空，则将当前信息按概率传播给它的所有粉丝
        else:
            for msg_id in nodes_screen[tweet_node]:
                if msg_id == -1: break
                # 真信息的情况
                if message_veracity[msg_id] == 1 and (random.random() <= true_retweet_prob):
                    for i in follower:
                        nodes_screen[i].pop()
                        nodes_screen[i].appendleft(msg_id)
                        message_popularity[msg_id] += 1
                # 假信息的情况
                elif message_veracity[msg_id] == -1 and (random.random() <= false_retweet_prob):
                    for i in follower:
                        nodes_screen[i].pop()
                        nodes_screen[i].appendleft(msg_id)
                        message_popularity[msg_id] += 1

        # 不同时间的真假信息传播规模分布
        if t in record_list:
            true_popularity, false_popularity = [], []
            for i in range(message_id):
                # if message_veracity.get(i) is None: continue
                if message_veracity[i] == 1:
                    true_popularity.append(message_popularity[i])
                elif message_veracity[i] == -1:
                    false_popularity.append(message_popularity[i])
            plt.figure(figsize=(8, 8))
            plot_distribution(true_popularity, 'CCDF', start=1, color='g', marker='s')
            plot_distribution(false_popularity, 'CCDF', start=1, color='r', marker='o')
            plt.axis([1, 1e6, 1e-5, 1])
            plt.legend(['TRUE', 'FALSE'])
            plt.tick_params(labelsize=20)
            plt.xlabel('Size', fontsize=30)
            plt.ylabel('CCDF', fontsize=30)
            plt.show()
        # 10000 11000 12000 13000
        if t+1 > 2000 and t<= 2000+1e5 and (t+1) % 1000 == 0:
            message_popularity_list.append(copy.deepcopy(message_popularity))
    #         fp.write(str(message_popularity))
    #         fp.write(',')
    # fp.write(']')
    # fp.close()
    print(message_id)
    return (message_popularity_list,message_veracity)







"""
# 一是信息数据与疾病数据是滞后相关的，本质上是疾病传播驱动了信息的传播
# 二是若已知未来疾病的发展情况可以推动未来的谣言分布，做好舆论引导和管控工作
# 如果放任谣言传播，则会处于超临界导致更多大规模的谣言出现而影响信息生态
# 三是疾病驱动下的多信息传播模型，及其理论求解，这部分很难，考虑平均场试试
# 重点放在疾病驱动信息传播，传播分布和临界转换，不要看重时间演化
"""

"""
首先，分析疾病传播和信息传播的关系发现滞后关系，把这种因素放入模型当中
其次，发现信息传播的三阶段传播特征，三个过程的幂律关系可以说明所有时期都不能放松对谣言的控制

多信息竞争传播过程
# 两个重要的参数分别是产生概率pt和转发概率pr，产生概率依赖于疫情的发展，而转发概率满足幂律分布
# 简单起见，网络结构可以考虑为无向网络处理，其实有向网络结果差不多的
# 网络拓扑结构考虑为随机规则网络和额SF网络，优先使用SF网络

# 传播过程
1. 每个时间步，随机选择一个节点进行操作
2. 节点以产生概率pt产生一条新信息存入自己的屏幕并传播给其邻居，其邻居也会更新自己的屏幕
3. 注意每个节点的屏幕的长度为L（可以假设为1），产生新信息则直接存入屏幕中，先进先出操作
3. 然后遍历节点屏幕中的所有信息，并且每个信息以传播概率pt传播给其邻居，同样更新邻居屏幕

网络规模的为10000，幂指数为2.5，平均度为10，传播时间为40N，实验重复10次
整个过程就是以概率pn产生信息，以概率pt传播信息，并且更新节点的屏幕
跳过初始的生成阶段，pn不会影响结果的

要么做模型驱动，要么做数据驱动，疾病发展不可准确预测，但是可以拟合

不同时期的谣言传播具有不同的特征，前期集中于疫情，后期集中于恢复

注意这里强调的是谣言数量，而不是谣言的传播量，延迟传播模型
可以用SIR模型来描述疾病发展的情况，然后反应到谣言传播模型上

"""

if __name__ == "__main__":
    print("model")
    (size_list,veracity) = model()







    # mean_y = [0]
    # std_y = [0]
    # time = np.array([1000*i for i in range(101)])
    # for size in size_list:
    #     size = np.array(size)
    #     mean_y.append(np.mean(size))
    #     std_y.append(np.std(size))
    # mean_y = np.array(mean_y)
    # std_y = np.array(std_y)
    # plt.figure(figsize=(8,8))
    # plt.plot(time,mean_y,'g',marker='o',markersize=8,linewidth=3)
    # plt.fill_between(time,mean_y+std_y,mean_y-std_y,facecolor='9', alpha=0.3)
    # plt.tick_params(labelsize=20)
    # plt.xlabel('time', fontsize=30)
    # plt.ylabel('cascades_size', fontsize=30)
    # plt.show()
    # np.array(a).T
    # first = size_list[0]
    # for i in range(len(size_list[0])):
    #     if size_list[0][i] == 0:
    #         end = i
    #         break
    # # 第end消息开始就不要了
    # 对size_list移动 100行 ，每行20000个数据
    a = np.array(size_list).T  # 20000行，每行100个数据
    a_list = []
    for temp in a:
        if temp[-1]==0:
            continue
        ret = list(filter(lambda x:x>0,temp))
        ret += [ret[-1]]*(len(temp)-len(ret))
        a_list.append(ret)
    filter_size_list = np.array(a_list).T  # 100行 11266列
    mean_T = []
    std_T = []
    mean_F = []
    std_F = []
    for size in filter_size_list:# size 是一个时刻所有消息的快照
        size_T = []
        size_F = []
        for i in range(len(size)):
            if veracity[i] == 1:
                size_T.append(size[i])
            if veracity[i] == -1:
                size_F.append(size[i])
        mean_T.append(np.mean(np.array(size_T)))
        std_T.append(np.std(np.array(size_T)))
        mean_F.append(np.mean(np.array(size_F)))
        std_F.append(np.std(np.array(size_F)))
    # mean_T = []
    # std_T = []
    # mean_F = []
    # std_F = []
    # for size in size_list:
    #     if size[0] == 0:
    #         break
    #     size_T = []
    #     size_F = []
    #     for i in range(len(size)):
    #         if veracity[i] == 1:
    #             size_T.append(size[i])
    #         if veracity[i] == -1:
    #             size_F.append(size[i])
    #     mean_T.append(np.mean(np.array(size_T)))
    #     std_T.append(np.std(np.array(size_T)))
    #     mean_F.append(np.mean(np.array(size_F)))
    #     std_F.append(np.std(np.array(size_F)))


    mean_T = np.array(mean_T)
    std_T = np.array(std_T)
    mean_F = np.array(mean_F)
    std_F = np.array(std_F)
    time = [1000*i for i in range(1,101)]

    plt.figure(figsize=(8,8))
    plt.plot(time,mean_T,'g',marker='o',markersize=8,linewidth=3)
    plt.fill_between(time,mean_T+std_T,mean_T-std_T,facecolor='g', alpha=0.3)
    plt.plot(time,mean_F,'r',marker='s',markersize=8,linewidth=3)
    plt.fill_between(time, mean_F + std_F, mean_F - std_F, facecolor='r', alpha=0.3)
    # plt.ylim([-2000,5000])
    plt.legend(['TRUE', 'FALSE'])
    plt.tick_params(labelsize=20)
    plt.xlabel('time', fontsize=30)
    plt.ylabel('cascades_size', fontsize=30)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(time, mean_T, 'g', marker='o', markersize=8, linewidth=3)
    # plt.fill_between(time, mean_T + std_T, mean_T - std_T, facecolor='g', alpha=0.3)
    plt.plot(time, mean_F, 'r', marker='s', markersize=8, linewidth=3)
    # plt.fill_between(time, mean_F + std_F, mean_F - std_F, facecolor='r', alpha=0.3)
    # plt.ylim([-2000,5000])
    plt.legend(['TRUE', 'FALSE'])
    plt.tick_params(labelsize=20)
    plt.xlabel('time', fontsize=30)
    plt.ylabel('cascades_size', fontsize=30)
    plt.show()









    # a = np.array(size_list).T
    # plt.figure(figsize=(8, 8))
    # for i in range(10000):
    #     # if veracity[i] == 1:
    #     #     plt.plot(a[i],'g')
    #     if veracity[i] == -1:
    #         plt.plot(a[i],'r')
    #
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # for i in range(10000):
    #     if veracity[i] == 1:
    #         plt.plot(a[i],'g')
    #     # if veracity[i] == -1:
    #     #     plt.plot(a[i], 'r')
    #
    # plt.show()
    #
    # mean_T = []
    # mean_F = []
    # for size in size_list:
    #     size_T = []
    #     size_F = []
    #     for i in range(len(size)):
    #         if size[i] == 0:
    #             break
    #         else:
    #             if veracity[i] == 1:
    #                 size_T.append(size[i])
    #             if veracity[i] == -1:
    #                 size_F.append(size[i])
    #     mean_T.append(np.mean(np.array(size_T)))
    #     mean_F.append(np.mean(np.array(size_F)))
    # plt.figure(figsize=(8,8))
    # plt.plot(mean_T,'g')
    # plt.plot(mean_F,'r')
    # plt.show()


