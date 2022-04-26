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
    # print(
    #     f'Brute Force Result: Minimax distance is {global_min_dist:.2f}, corresponding vertices are {components}')
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

def initialize_ex_map():
    vertices = set()
    vertices.add(Vertex(22,90))
    vertices.add(Vertex(25,132))
    vertices.add(Vertex(7,131))
    vertices.add(Vertex(33,97))
    vertices.add(Vertex(28,96))
    vertices.add(Vertex(25,98))
    vertices.add(Vertex(22,90))
    vertices.add(Vertex(19,87))
    vertices.add(Vertex(30,121))
    vertices.add(Vertex(25,118))
    vertices.add(Vertex(22,119))
    vertices.add(Vertex(17,110))
    vertices.add(Vertex(16,116))
    vertices.add(Vertex(24,134))
    vertices.add(Vertex(13,132))
    vertices.add(Vertex(6,122))
    vertices.add(Vertex(15,174))
    vertices.add(Vertex(7,169))

    return list(vertices)

def loblaw():
    vertices = set()
    vertices.add(Vertex(22,90))
    vertices.add(Vertex(25,132))
    vertices.add(Vertex(7,131))

    return list(vertices)

def microcenter_all():
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

def microcenter_hq():
    vertices = set()
    vertices.add(Vertex(48,139))

    return list(vertices)

def plot_connect_2_points(a_list,b_list, color, linestyle):
    x_values = [a_list[0], b_list[0]]
    y_values = [a_list[1], b_list[1]]
    plt.plot(x_values, y_values, color=color, linestyle=linestyle)
    return

def plot_connection_lines(center, points_set, color, linestyle):
    center=center[0].points()
    for a in points_set:
        plot_connect_2_points(center, a.points(),color, linestyle)
    return

def frq(vertices):
    x_dict = defaultdict(lambda:0)
    y_dict = defaultdict(lambda:0)

    for v in vertices:
        raw_x, raw_y = v.points()
        x = find_bucket(raw_x)
        y = find_bucket(raw_y)
        x_dict[x] += 1
        y_dict[y] += 1
    
    x_fl = sorted([[k,v] for k, v in x_dict.items()], key = lambda x:x[1], reverse=True)
    y_fl = sorted([[k,v] for k, v in y_dict.items()], key = lambda x:x[1], reverse=True)
    # print(x_dict)
    # print(y_dict)
    # print([(x_fl[0][0])*10+5, (y_fl[0][0])*10+5])
    return Vertex((x_fl[0][0])*10+5, (y_fl[0][0])*10+5)

def find_bucket(num):
    if num < 10:
        return 0
    elif num == 200:
        return 19
    else:
        return num//10

def dist(a,b):
    x1, y1 = a.points()
    x2, y2 = b.points()
    x = x1-x2
    y = y1-y2
    return math.sqrt((x**2+y**2))

def find_closet(vertices, point):
    min_dist = 1000000000
    target = []
    for v in vertices:
        if dist(point, v)<min_dist:
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
    return [chosen] , total

if __name__ == '__main__':
    #vertices = initialize(20)
    # vertices = initialize_ex_map()
    # real = loblaw()
    # brute_force_ans = k_center(vertices, 3)
    # ans = greedy(set(vertices), 3)

    #Microcenter
    vertices = microcenter_all()
    ans, dis = greedy1(vertices)
    real = microcenter_hq()

    # im=plt.imread("usa_map.png")
    # fig, ax = plt.subplots()
    # im = ax.imshow(im, extent=[0, 130, 0, 100])

    plot_points(vertices, 'o', 'blue')
    plot_points(real, '*', 'green')
    plot_points(ans, '^', 'yellow')
    # print(dis)

    # plot_connection_lines(real, vertices, color='r', linestyle=':')
    # plot_connection_lines(ans, vertices, color='y', linestyle=':')
    plt.show()
