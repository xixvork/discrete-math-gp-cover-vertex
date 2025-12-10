import random
import math
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import sys
from io import StringIO

class Graph:
    
    def __init__(self, n, delta):
        
        if not 0 <= delta <= 1:
            raise ValueError("Щільність (delta) має бути в діапазоні [0, 1]")
        
        self.n = n
        self.delta = delta
        self.adj_list = [[] for _ in range(self.n)]
        self.adj_matrix = [[0] * self.n for _ in range(self.n)]
        self.vertex_cover = set()
        
    def _generate(self):
        self.adj_list = [[] for _ in range(self.n)]
        self.adj_matrix = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.random() < self.delta:
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
                    self.adj_matrix[i][j] = 1
                    self.adj_matrix[j][i] = 1

    def generate_vertex_cover_from_list(self):
        edges = []
        for u, neighbors in enumerate(self.adj_list):
            for v in neighbors:
                if u < v:
                    edges.append((u, v))
        
        random.shuffle(edges)
        covered_vertices = set()

        for u, v in edges:
            if u not in covered_vertices and v not in covered_vertices:
                covered_vertices.add(u)
                covered_vertices.add(v)
        
        self.vertex_cover = covered_vertices

    def generate_vertex_cover_from_matrix(self):
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj_matrix[i][j] == 1:
                    edges.append((i, j))
        
        random.shuffle(edges)
        covered_vertices = set()

        for u, v in edges:
            if u not in covered_vertices and v not in covered_vertices:
                covered_vertices.add(u)
                covered_vertices.add(v)
        
        self.vertex_cover = covered_vertices


def make_experiment(n_vertices, density, repeats, method='matrix') -> list[float]:
    res = [] 
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        for _ in range(repeats):
            my_graph = Graph(n_vertices, density)
            my_graph._generate()

            t1 = time.time()
            if method == 'list':
                my_graph.generate_vertex_cover_from_list()
            else:
                my_graph.generate_vertex_cover_from_matrix()
            t2 = time.time()
            res.append(t2 - t1)
    finally:
        sys.stdout = original_stdout
        
    return res

DENSITY = 0.1
REPEATS = 50
SIZES = [20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
METHODS = ['list', 'matrix']
OUTPUT_FILENAME = "methods_comparison_results.csv"

results_data = {}

for method in METHODS:
    method_times = {}
    for size in SIZES:
        print(f"Проведення експерименту: метод={method}, n={size}, щільність={DENSITY}, повторень={REPEATS}")
        
        times = make_experiment(n_vertices=size, density=DENSITY, repeats=REPEATS, method=method)
        
        avg_time = sum(times) / len(times)
        method_times[size] = avg_time
        
    results_data[method] = method_times

df = pd.DataFrame(results_data)

df_transposed = df.transpose()
df_transposed.index.name = "Метод"

df_transposed.to_csv(OUTPUT_FILENAME, float_format='%.7f')

pd.options.display.float_format = '{:.7f}'.format

print(f"\n--- Підсумкова таблиця результатів ({REPEATS} повторень) ---")
print(df_transposed)
print(f"\nЕксперименти завершено. Таблицю збережено у файл '{OUTPUT_FILENAME}'.")
