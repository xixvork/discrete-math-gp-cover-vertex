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
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.random() < self.delta:
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
                    self.adj_matrix[i][j] = 1
                    self.adj_matrix[j][i] = 1

    def generate_vertex_cover_from_list(self):
        print("\n--- Генерація вершинного покриття (зі списку суміжності) ---")

        edges = []
        for u, neighbors in enumerate(self.adj_list):
            for v in neighbors:
                if u < v:
                    edges.append((u, v))
        
        random.shuffle(edges)

        maximal_matching = set()
        covered_vertices = set()

        for u, v in edges:
            if u not in covered_vertices and v not in covered_vertices:
                maximal_matching.add((u, v))
                covered_vertices.add(u)
                covered_vertices.add(v)
        
        new_vertex_cover = set()
        for u, v in maximal_matching:
            new_vertex_cover.add(u)
            new_vertex_cover.add(v)
            
        self.vertex_cover = new_vertex_cover
        print(f"Знайдене максимальне парування (M): {maximal_matching}")
        print(f"Згенероване вершинне покриття (C): {sorted(list(self.vertex_cover))}")

    def generate_vertex_cover_from_matrix(self):
        print("\n--- Генерація вершинного покриття (з матриці суміжності) ---")

        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj_matrix[i][j] == 1:
                    edges.append((i, j))
        
        random.shuffle(edges)

        maximal_matching = set()
        covered_vertices = set()

        for u, v in edges:
            if u not in covered_vertices and v not in covered_vertices:
                maximal_matching.add((u, v))
                covered_vertices.add(u)
                covered_vertices.add(v)
        
        new_vertex_cover = set()
        for u, v in maximal_matching:
            new_vertex_cover.add(u)
            new_vertex_cover.add(v)
            
        self.vertex_cover = new_vertex_cover
        print(f"Знайдене максимальне парування (M): {maximal_matching}")
        print(f"Згенероване вершинне покриття (C): {sorted(list(self.vertex_cover))}")

    def display_adjacency_list(self):
        print("--- Список суміжності ---")
        if not self.adj_list:
            print("Граф порожній.")
            return
        for i, neighbors in enumerate(self.adj_list):
            print(f"Вершина {i}: -> {sorted(neighbors)}")

    def display_adjacency_matrix(self):
        print("\n--- Матриця суміжності ---")
        if not self.adj_matrix:
            print("Граф порожній.")
            return
        for row in self.adj_matrix:
            print(" ".join(map(str, row)))

    def visualize(self):
        if self.n == 0:
            print("Граф порожній, нічого візуалізувати.")
            return
        
        positions = {i: (math.cos(2 * math.pi * i / self.n), math.sin(2 * math.pi * i / self.n)) for i in range(self.n)}
        fig, ax = plt.subplots(figsize=(8, 8))

        for i, neighbors in enumerate(self.adj_list):
            for neighbor in neighbors:
                if i < neighbor:
                    x_coords = [positions[i][0], positions[neighbor][0]]
                    y_coords = [positions[i][1], positions[neighbor][1]]
                    ax.plot(x_coords, y_coords, color='gray', zorder=1)

        node_colors = ['red' if i in self.vertex_cover else 'blue' for i in range(self.n)]
        
        ax.scatter([pos[0] for pos in positions.values()], 
                   [pos[1] for pos in positions.values()], 
                   s=600, 
                   c=node_colors,
                   edgecolors='black', 
                   zorder=2)
        
        for i, pos in positions.items():
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontsize=10, weight='bold')

        ax.set_aspect('equal', adjustable='box')
        plt.title("Візуалізація графа з вершинним покриттям (червоні)")
        plt.box(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

def make_experiment(n_vertices, density, repeats, method='matrix') -> list[float]:
    res = [] 
    while repeats > 0:
        repeats -= 1

        my_graph = Graph(n_vertices, density)
        my_graph._generate()

        t1 = time.time()
        if method == 'list':
            my_graph.generate_vertex_cover_from_list()
        else:
            my_graph.generate_vertex_cover_from_matrix()
        t2 = time.time()

        res.append(t2 - t1)
    return res

if not os.path.exists("experiments"):
    os.makedirs("experiments")

densities = [0.05, 0.1, 0.2, 0.3, 0.5]
sizes = [20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
repeats = 50

results = {density: {} for density in densities}

for density in densities:
    for size in sizes:
        print(f"Проведення експерименту: n={size}, delta={density}")
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        times = make_experiment(n_vertices=size, density=density, repeats=repeats, method='list')
        
        sys.stdout = old_stdout
        output = captured_output.getvalue()

        filename = f"experiments/list_n{size}_d{density}.txt"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(output)
            avg_time = sum(times) / len(times)
            f.write(f"\nСередній час виконання: {avg_time:.10f} секунд\n")
            
        results[density][size] = avg_time

df = pd.DataFrame(results)
df.index.name = "Розмір графа"
df_to_save = df.copy()
for col in df_to_save.columns:
    df_to_save[col] = df_to_save[col].map('{:.7f}'.format)
df_to_save.to_csv("list_experiment_results.csv")
pd.options.display.float_format = '{:.7f}'.format

print("\n--- Результати експериментів ---")
print(df)
print("\nУсі експерименти завершено. Результати збережено в папці 'experiments/' та у файлі 'list_experiment_results.csv'.")
