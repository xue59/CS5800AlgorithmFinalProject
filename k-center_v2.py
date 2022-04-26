import random
from itertools import combinations

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


def plot_points(vertexes, color):
    points = []
    for v in vertexes:
        points.append(v.points())

    xs = [x for [x, _] in points]
    ys = [y for [_, y] in points]
    plt.plot(xs, ys, 'bo', color=color)


def initialize():
    min = 0
    max = 200
    vertices = set()
    for i in range(0, 100):
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
        f'Brute Force Result: Minimax distance is {global_min_dist}, corresponding vertices are {components}')
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

    print(f'Greedy Solution Result: Minmax distance is {global_max_dist}, corresponding vertices are {G}')
    return G


if __name__ == '__main__':
    vertices = initialize()

    brute_force_ans = k_center(vertices, 1)
    ans = greedy(set(vertices), 1)

    plot_points(vertices, 'blue')
    #plot_points(ans, 'red')
    plot_points(brute_force_ans, 'yellow')
    plt.show()
