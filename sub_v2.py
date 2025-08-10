import numpy as np
import os
import random
import math
import heapq

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def orientation(a, b, c):
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])  # 外積
    return sign(cross) # a, b, c の順番の向き(1: 反時計周り, -1: 反時計回り, 0: 一直線)

def segments_intersect(p1, p2, q1, q2):
    if p1==q1 or p1==q2 or p2==q1 or p2==q2:
        return False
    if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or
        max(q1[0], q2[0]) < min(p1[0], p2[0]) or
        max(p1[1], p2[1]) < min(q1[1], q2[1]) or
        max(q1[1], q2[1]) < min(p1[1], p2[1])):
        return False
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    return (o1 * o2 <= 0) and (o3 * o4 <= 0)

def parse_input():
    n, m, k = list(map(int, input().strip().split()))

    d_pos = []
    for _ in range(n):
        x, y = list(map(int, input().strip().split()))
        d_pos.append((x, y))

    s_pos = []
    for _ in range(m):
        x, y = list(map(int, input().strip().split()))
        s_pos.append((x, y))

    prob = []
    for _ in range(k):
        p = list(map(float, input().strip().split()))
        prob.append(p)
    return (n, m, k, d_pos, s_pos, prob)

n, m, k, d_pos, s_pos, prob = parse_input()

def sorter(in_prob, i):
    p = np.array(prob[i])
    out_prob1 = in_prob * p
    out_prob2 = in_prob * (1-p)
    return (out_prob1, out_prob2)

def opt_processor(out_prob):
    max_i = np.argmax(out_prob)
    max_prob = out_prob[max_i]
    
    return (max_i, max_prob)

def calc_opt_score(out_probs):
    probs = [0.0 for _ in range(n)]
    to_processors = [0 for _ in range(n)]
    for out_prob in out_probs:
        (max_i, max_prob) = opt_processor(out_prob)
        probs[max_i] += max_prob
        to_processors[max_i] += 1
    score = calc_score(probs)

    return (score, probs, to_processors)

def calc_score(probs):
    return round(1_000_000_000 * (n - sum(probs))/n)

in_prob = np.ones(n)
out_probs = [in_prob]
score, probs, to_processors = calc_opt_score(out_probs)

def opt_sorter2(in_prob):
    opt_i = 0
    opt_prob = 0.0
    opt_processors = [0, 0]
    for i in range(k):
        out_probs = sorter(in_prob, i)
        sum_prob = 0.0
        processors = []
        for out_prob in out_probs:
            (max_i, max_prob) = opt_processor(out_prob)
            sum_prob += max_prob
            processors.append(max_i)
        if opt_prob < sum_prob:
            opt_prob = sum_prob
            opt_i = i
            opt_processors = processors
    return (opt_i, opt_prob, opt_processors)

# 基準点
entrance = (0, 5000)
# 距離を計算する関数
def calculate_distance(point):
    return math.sqrt((point[0] - entrance[0])**2 + (point[1] - entrance[1])**2)

# 2分木関連
def parent(i):
    if i==0:
        return None
    return (i-1)//2

def left(i):
    return 2*i+1

def right(i):
    return 2*i+2

def calc_dist(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def connectable(p1, p2):
    for i in range(len(edges)):
        if edges[i] is None:
            continue
        (q1, q2) = edges[i]
        if segments_intersect(p1, p2, q1, q2):
            return False
    return True

def dfs(node_i, pos_i):
    if node_i == 0:
        frm = (0, 5000)
    else:
        frm = node_pos[parent(node_i)]
    if node_i < len(sorters):  # 分別器の配置
        start_i = pos_i
        for (ii, (_, i)) in enumerate(pos_data[start_i:]):
            to = s_pos[i]
            if not connectable(frm, to):
                continue
            ans_s[i] = [sorters[node_i]]
            edges[node_i] = (frm, to)
            node_pos[node_i] = to
            indexes[node_i] = i
            if node_i == 0:
                ans_start[0] = n+i
            else:
                parent_i = indexes[parent(node_i)]
                ans_s[parent_i].append(n+i)
            if dfs(node_i+1, start_i+ii+1):
                return True
    else:  # 処理装置の配置
        # 接続したい装置が未配置なら、一番近いところから順に確認して、接続可能なところに配置して、接続する。接続済みの場所は不可
        # すでに配置済みなら、接続できるか確認する。できなければ、接続できるところに接続する
        p = processors[node_i-len(sorters)]
        if p in ans_d:  # 配置済みの場合
            ii = ans_d.index(p)
            to = d_pos[ii]
            if connectable(frm, to):
                edges[node_i] = (frm, to)
                node_pos[node_i] = to
                parent_i = indexes[parent(node_i)]
                ans_s[parent_i].append(ii)
                if node_i+1 == len(sorters)+len(processors):
                    return True
                elif dfs(node_i+1, -1):
                    return True
            else:
                # 接続できるところに接続する
                d_dist = []
                for (i, pos) in enumerate(d_pos):
                    dist = calc_dist(frm, pos)
                    d_dist.append((dist, i))
                d_dist.sort()
                for (_, ii) in d_dist:
                    to = d_pos[ii]
                    if connectable(frm, to):
                        edges[node_i] = (frm, to)
                        node_pos[node_i] = to
                        parent_i = indexes[parent(node_i)]
                        ans_s[parent_i].append(ii)
                        if node_i+1 == len(sorters)+len(processors):
                            return True
                        elif dfs(node_i+1, -1):
                            return True
        else:  # 未配置の場合
            d_dist = []
            for (i, pos) in enumerate(d_pos):
                if ans_d[i] is not None:
                    continue
                dist = calc_dist(frm, pos)
                d_dist.append((dist, i))
            d_dist.sort()
            for (_, ii) in d_dist:
                to = d_pos[ii]
                if connectable(frm, to):
                    ans_d[ii] = p
                    edges[node_i] = (frm, to)
                    node_pos[node_i] = to
                    parent_i = indexes[parent(node_i)]
                    ans_s[parent_i].append(ii)
                    if node_i+1 == len(sorters)+len(processors):
                        return True
                    elif dfs(node_i+1, -1):
                        return True
    return False  # 最後まで接続できたものなし

def solve(sorter_cnt):
    # sortersの計算
    for i in range(sorter_cnt):
        in_prob = out_probs.pop(0)
        (opt_i, opt_prob, opt_processors) = opt_sorter2(in_prob)
        ps = sorter(in_prob, opt_i)
        out_probs.extend(ps)
        sorters.append(opt_i)
        if len(processors) > 0:
            processors.pop(0)
        processors.extend(opt_processors)
        score, probs, to_processors = calc_opt_score(out_probs)

    sum_prob = 0.0
    processor_cnt = {i: 0 for i in range(n)}
    processor_probs = []
    for i, p in enumerate(processors):
        sum_prob += out_probs[i][p]
        processor_cnt[p] += 1
        processor_probs.append((out_probs[i][p], i, p))
    processor_probs.sort(reverse=True)

    # 配置処理
    for (i, pos) in enumerate(s_pos):
        distance = calculate_distance(pos)
        pos_data.append((distance, i))
    pos_data.sort()

    dfs(0, 0)

    # 回答の漏れ修正
    not_exists = []
    for i in range(n):
        if i not in ans_d:
            not_exists.append(i)
    for (i, di) in enumerate(ans_d):
        if di is None:
            ans_d[i] = not_exists.pop()

    for (i, a) in enumerate(ans_s):
        if ans_s[i][0] == -1:
            continue
        if ans_s[i][1] == -1 and ans_s[i][2] == -1:
            xy = s_pos[i]
            for (ii, pos) in enumerate(d_pos):
                if connectable(xy, pos):
                    ans_s[i][1] = ii
                    ans_s[i][2] = ii
                    break
        elif ans_s[i][1] == -1:
            ans_s[i][1] = ans_s[i][2]
        elif ans_s[i][2] == -1:
            ans_s[i][2] = ans_s[i][1]

    # 出力        
    print(*ans_d)
    print(ans_start[0])
    for a in ans_s:
        print(*a)

sorter_cnt = 1
ans_d = [None for _ in range(n)]
ans_s = [[-1] for _ in range(m)]
ans_start = [-1]
indexes = [-1 for _ in range(m)]
edges = [None for _ in range(sorter_cnt*2+1)]
node_pos = [None for _ in range(sorter_cnt*2+1)]
pos_data = []
sorters = []
processors = []
solve(sorter_cnt)
