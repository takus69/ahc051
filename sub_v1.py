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

def parse_input(filename):
    with open(os.path.join('in', filename), 'r') as file:
        n, m, k = list(map(int, file.readline().strip().split()))
        
        d_pos = []
        for _ in range(n):
            x, y = list(map(int, file.readline().strip().split()))
            d_pos.append((x, y))
        
        s_pos = []
        for _ in range(m):
            x, y = list(map(int, file.readline().strip().split()))
            s_pos.append((x, y))
        
        prob = []
        for _ in range(k):
            p = list(map(float, file.readline().strip().split()))
            prob.append(p)
    return (n, m, k, d_pos, s_pos, prob)

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

sorters = []
processors = []
for i in range(n-1):
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
    for (q1, q2) in edges:
        if segments_intersect(p1, p2, q1, q2):
            return False
    return True

# ヒープ (優先度付きキュー) を作成
que = []
for (i, pos) in enumerate(s_pos):
    distance = calculate_distance(pos)
    heapq.heappush(que, (distance, i))

# 近い順に取得
ans_d = [None for _ in range(n)]
ans_s = [[-1] for _ in range(m)]
# 最初の配置
(_, i) = heapq.heappop(que)
ans_start = n+i
now_i = 0
ans_s[i] = [sorters[now_i]]
xy = s_pos[i]
edges = [((0, 5000), xy)]
done_cnt = 1
indexes = [i]

# sorter の配置
while True:
    # 左の子
    (_, i) = heapq.heappop(que)
    while not connectable(xy, s_pos[i]):
        (_, i) = heapq.heappop(que)
    v1 = i
    # 接続
    ans_s[v1] = [sorters[left(now_i)]]
    ans_s[indexes[now_i]].append(n+v1)
    edges.append((xy, s_pos[v1]))
    indexes.append(v1)
    done_cnt += 1
    if done_cnt == len(sorters):
        break
    
    # 右の子
    (_, i) = heapq.heappop(que)
    while not connectable(xy, s_pos[i]):
        (_, i) = heapq.heappop(que)
    v2 = i
    # 接続
    ans_s[v2] = [sorters[right(now_i)]]
    ans_s[indexes[now_i]].append(n+v2)
    edges.append((xy, s_pos[v2]))
    indexes.append(v2)
    done_cnt += 1
    if done_cnt == len(sorters):
        break
    
    now_i += 1
    xy = s_pos[indexes[now_i]]
    
# processor の接続
for (_, i, p) in processor_probs:
    base_i = len(sorters)
    now_i = base_i + i
    parent_i = parent(now_i)
    si = indexes[parent_i]
    while len(ans_s[si]) < 3:
        ans_s[si].append(-1)
    if left(parent_i) == now_i:
        child_i = 1
    else:
        child_i = 2
    
    di = -1
    xy = s_pos[indexes[parent_i]]
    if p in ans_d:
        ii = ans_d.index(p)
        if connectable(xy, d_pos[ii]):
            di = ii
    else:
        que = []
        for (i, pos) in enumerate(d_pos):
            if ans_d[i] is not None:
                continue
            dist = calc_dist(xy, pos)
            heapq.heappush(que, (dist, i))
        (dd, i) = heapq.heappop(que)
        while not connectable(xy, d_pos[i]):
            if len(que)==0:
                di = None
                break
            (dd, i) = heapq.heappop(que)
        di = i
    
    if di != -1 and connectable(xy, d_pos[di]):
        ans_s[si][child_i] = di
        ans_d[di] = p
        edges.append((xy, d_pos[di]))

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

        
print(*ans_d)
print(ans_start)
for a in ans_s:
    print(*a)