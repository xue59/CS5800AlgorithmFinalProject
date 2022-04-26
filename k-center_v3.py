import random
from itertools import combinations
from collections import defaultdict

import math
import numpy as np
import pylab as plt

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%s,%s)" % (self.x, self.y)

    def __repr__(self):
        return "(%s,%s)" % (self.x, self.y)

    def points(self):
        return [self.x, self.y]


def plot_points(vertexes, marker, color):
    points = []
    for v in vertexes:
        points.append(v.points())

    xs = [x for [x, _] in points]
    ys = [y for [_, y] in points]
    if marker == 'o':
        size = 10
    elif marker == '*':
        size = 8
    else:
        size = 5
    plt.plot(xs, ys, linestyle="", marker=marker, color=color, markersize=size)


def initialize(n):
    min = 0
    max = 200
    vertices = set()
    for i in range(0, n):
        vertices.add(Vertex(random.randint(min, max), random.randint(min, max)))
    return list(vertices)


def min_distance(v_list, y):
    min_dist = math.inf
    for vertex in v_list:
        tmp_dist = math.sqrt((vertex.x - y.x) ** 2 + (vertex.y - y.y) ** 2)
        min_dist = min(tmp_dist, min_dist)
    return min_dist


def k_center(vertices, k):
    components = []
    global_min_dist = 10000.
    for indices in combinations(range(len(vertices)), k):
        tmp_vertices = [vertices[i] for i in indices]
        tmp_max_dist = 0.
        for i in range(len(vertices)):
            if i not in indices:
                tmp_max_dist = max(tmp_max_dist, min_distance(tmp_vertices, vertices[i]))
        if tmp_max_dist < global_min_dist:
            components = tmp_vertices
            global_min_dist = tmp_max_dist
    print(
        f'Brute Force Result: Minimax distance is {global_min_dist:.2f}, corresponding vertices are {components}')
    return components


def distance(vertices, picked):
    min_v = None
    real_max = 0
    real_max_v = None
    for v in vertices:
        min = np.inf
        for p in picked:
            d = np.sqrt((v.x - p.x) ** 2 + (v.y - p.y) ** 2)
            if d < min:
                min = d
                min_v = v
        if min > real_max:
            real_max = min
            real_max_v = min_v

    return real_max_v


def greedy(vertices, k):
    # min_dist = 0
    # pool = []
    # for i in range(0, len(vertices)):
    G = []
    pick = random.sample(vertices, 1)[0]
    vertices.remove(pick)
    G.append(pick)
    while len(G) < k:
        max = distance(vertices, G)
        vertices.remove(max)
        G.append(max)

    vertices = list(vertices)
    global_max_dist = 0.
    for v in vertices:
        tmp_dist = min_distance(G, v)
        if tmp_dist > global_max_dist:
            global_max_dist = tmp_dist

        # if min_dist>global_max_dist:
        #     min_dist = global_max_dist
        #     pool = G[:]
        #     global_max_dist = 10000

    print(f'Greedy Solution Result: Minmax distance is {global_max_dist:.2f}, corresponding vertices are {G}')
    return G

def zack_microcenter_map():
    vertices = set()
    vertices.add(Vertex(8.7,39.3))
    vertices.add(Vertex(66.6, 20.2))
    vertices.add(Vertex(61, 28.2))
    vertices.add(Vertex(43, 54))
    vertices.add(Vertex(67, 50))

    vertices.add(Vertex(77, 49))
    vertices.add(Vertex(70.5, 74.4))
    vertices.add(Vertex(83, 60))
    vertices.add(Vertex(83.7, 59))
    vertices.add(Vertex(94, 66.3))
    vertices.add(Vertex(98.4, 60.8))
    vertices.add(Vertex(93.8, 34.2))
    vertices.add(Vertex(95, 34.5))
    vertices.add(Vertex(121, 72))
    vertices.add(Vertex(91.6, 52.3))
    vertices.add(Vertex(96.1, 56.5))
    vertices.add(Vertex(95.9, 56.3))
    vertices.add(Vertex(109.4, 54))
    vertices.add(Vertex(110, 57))
    vertices.add(Vertex(112, 57.6))
    vertices.add(Vertex(113, 59))

    vertices.add(Vertex(115, 64.3))
    vertices.add(Vertex(115.5, 63.9))
    vertices.add(Vertex(114.8, 64))
    vertices.add(Vertex(115.2, 64.2))

    return list(vertices)
def initialize_tim_lob_law_map():
    vertices = set()
    vertices.add(Vertex(3.7, 32.3))
    vertices.add(Vertex(8.9, 38.6))
    vertices.add(Vertex(6.8, 54.6))
    vertices.add(Vertex(19, 60))
    vertices.add(Vertex(17.6, 50))
    vertices.add(Vertex(21.8, 43))
    vertices.add(Vertex(48, 35.5))
    vertices.add(Vertex(50.7, 52.6))
    vertices.add(Vertex(47.5, 35))
    vertices.add(Vertex(37.3, 27))
    vertices.add(Vertex(44, 24.5))
    vertices.add(Vertex(67, 41.6))
    vertices.add(Vertex(67, 20.2))
    vertices.add(Vertex(52, 3.9))
    vertices.add(Vertex(42, 44.36))
    ##distribution center
    vertices.add(Vertex(8.9, 38.6))
    vertices.add(Vertex(2.3, 47.3))
    vertices.add(Vertex(62, 43))

    return list(vertices)

def loblaw():
    loblaw_vertices = set()
    loblaw_vertices.add(Vertex(8.9, 38.6))
    loblaw_vertices.add(Vertex(2.3, 47.3))
    loblaw_vertices.add(Vertex(62, 43))
    return list(loblaw_vertices)

def plot_connect_2_points(a_list,b_list, color, linestyle):
    x_values = [a_list[0], b_list[0]]
    y_values = [a_list[1], b_list[1]]
    plt.plot(x_values, y_values, color=color, linestyle=linestyle)
    return

def plot_connection_lines(center, points_set, color, linestyle):
    if isinstance(center, Vertex):
        center = center.points()
    else:
        center=center[0].points()
    for a in points_set:
        plot_connect_2_points(center, a.points(),color, linestyle)
    return


def tim_microcenter_all():
    vertices = set()
    vertices.add(Vertex(71,99))
    vertices.add(Vertex(26,2))
    vertices.add(Vertex(51,54))
    vertices.add(Vertex(47,93))
    vertices.add(Vertex(23,85))
    vertices.add(Vertex(10,91))
    vertices.add(Vertex(59,121))
    vertices.add(Vertex(57,120))
    vertices.add(Vertex(59,139))
    vertices.add(Vertex(55,146))
    vertices.add(Vertex(46,135))
    vertices.add(Vertex(24,136))
    vertices.add(Vertex(25,134))
    vertices.add(Vertex(49,140))
    vertices.add(Vertex(48,139))
    vertices.add(Vertex(44,163))
    vertices.add(Vertex(51,177))
    vertices.add(Vertex(51,176))

    return list(vertices)


def frq(vertices):
    x_dict = defaultdict(lambda: 0)
    y_dict = defaultdict(lambda: 0)

    for v in vertices:
        raw_x, raw_y = v.points()
        x = find_bucket(raw_x)
        y = find_bucket(raw_y)
        x_dict[x] += 1
        y_dict[y] += 1

    x_fl = sorted([[k, v] for k, v in x_dict.items()], key=lambda x: x[1], reverse=True)
    y_fl = sorted([[k, v] for k, v in y_dict.items()], key=lambda x: x[1], reverse=True)
    # print(x_dict)
    # print(y_dict)
    # print([(x_fl[0][0])*10+5, (y_fl[0][0])*10+5])
    return Vertex((x_fl[0][0]) * 10 + 5, (y_fl[0][0]) * 10 + 5)


def find_bucket(num):
    if num < 10:
        return 0
    elif num == 200:
        return 19
    else:
        return num // 10


def dist(a, b):
    x1, y1 = a.points()
    x2, y2 = b.points()
    x = x1 - x2
    y = y1 - y2
    return math.sqrt((x ** 2 + y ** 2))


def find_closet(vertices, point):
    min_dist = 1000000000
    target = []
    for v in vertices:
        if dist(point, v) < min_dist:
            min_dist = dist(v, point)
            target = v
    print(target)
    return target


def greedy1(vertices):
    chosen = find_closet(vertices, frq(vertices))
    vertices.remove(chosen)
    # print(vertices)
    total = 0
    for v in vertices:
        total += dist(v, chosen)
    return [chosen], total

def microcenter_hq():
    vertices = set()
    vertices.add(Vertex(96.1, 56.5)) #change to zack hq location

    return list(vertices)

def greedy_tim(vertices, k):
    origin = vertices.copy()
    min_dist = 100000000
    for i in range(len(origin)*k*1000):
        G = []
        pick = random.sample(origin, 1)[0]
        origin.remove(pick)
        G.append(pick)
        while len(G) < k:
            max = distance(origin, G)
            origin.remove(max)
            G.append(max)

        origin = list(origin)
        global_max_dist = 0.
        for v in origin:
            tmp_dist = min_distance(G, v)
            if tmp_dist > global_max_dist:
                global_max_dist = tmp_dist

        if global_max_dist<min_dist:
            min_dist = global_max_dist
            final = G[:]
        origin = vertices.copy()

    # print(f'Greedy Solution Result: Minmax distance is {global_max_dist:.2f}, corresponding vertices are {G}')
    return final

if __name__ == '__main__':
    #vertices = initialize(20)
    #vertices = initialize_tim_lob_law_map()
    #brute_force_ans = k_center(vertices, 3)
    #ans = greedy(set(vertices), 3)

    #im=plt.imread("usa_map.png")
    #fig, ax = plt.subplots()
    #im = ax.imshow(im, extent=[0, 130, 0, 100])

    #Microcenter
    # vertices = microcenter_all()
    #ans, dis = greedy1(vertices)
    #real = microcenter_hq()

    #loblaw
    # im = plt.imread("tim_loblaw_map.png")
    # fig, ax = plt.subplots()
    # im = ax.imshow(im, extent=[0, 80, 0, 65])
    #
    # vertices = initialize_tim_lob_law_map()
    # tim_loblaw_ans=greedy_tim(set(vertices), 3)
    # real = loblaw()
    # plot_points(vertices, 'o', 'red')
    # plot_points(real, '*', 'green')
    # plot_points(tim_loblaw_ans, '^', 'yellow')
    # plot_connection_lines(tim_loblaw_ans[0], vertices, color='y', linestyle=':')
    # plot_connection_lines(tim_loblaw_ans[1], vertices, color='y', linestyle=':')
    # plot_connection_lines(tim_loblaw_ans[2], vertices, color='y', linestyle=':')


    #US map & microcenter
    im=plt.imread("usa_map.png")
    fig, ax = plt.subplots()
    im = ax.imshow(im, extent=[0, 130, 0, 100])

    vertices = zack_microcenter_map()
    brute_force_ans = k_center(vertices, 1)
    ans_rand = greedy(set(vertices), 1)
    tim_ans=greedy_tim(vertices,1)
    plot_points(vertices, 'o', 'blue')
    plot_points(tim_ans, '*', 'red')
    plot_points(ans_rand, '^', 'yellow')

    #plot_points(ans_rand, '^', 'yellow')
    plot_connection_lines(tim_ans, vertices, color='r', linestyle=':')
    plot_connection_lines(ans_rand, vertices, color='y', linestyle=':')
    plt.show()
