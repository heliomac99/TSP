import numpy as np
import networkx as nx
from utils import read_tsplib
import sys

sys.setrecursionlimit(2000)  # Ajuste para instâncias maiores


import math
def graph_to_adj_matrix(graph):
    """
    Converte um grafo do NetworkX em uma matriz de adjacência.

    :param graph: Grafo do NetworkX
    :return: Matriz de adjacência (lista de listas)
    """
    nodes = list(graph.nodes)
    size = len(nodes)
    adj_matrix = np.full((size, size), float('inf'))  # Inicializa com infinito

    for i in nodes:
        for j in nodes:
            if i != j and graph.has_edge(i, j):
                adj_matrix[i][j] = graph[i][j]['weight']

    return adj_matrix.tolist()

# TSP com Branch and Bound
class BranchAndBound:
    def __init__(self, adj_matrix):
        """
        Inicializa o algoritmo Branch and Bound para resolver o TSP.

        :param adj_matrix: Matriz de adjacência representando o problema
        """
        self.adj = adj_matrix
        self.N = len(adj_matrix)
        self.final_path = [None] * (self.N + 1)
        self.final_res = float('inf')

    def copy_to_final(self, curr_path):
        self.final_path[:self.N + 1] = curr_path[:]
        self.final_path[self.N] = curr_path[0]

    def first_min(self, i):
        min_val = float('inf')
        for k in range(self.N):
            if self.adj[i][k] < min_val and i != k:
                min_val = self.adj[i][k]
        return min_val

    def second_min(self, i):
        first, second = float('inf'), float('inf')
        for j in range(self.N):
            if i == j:
                continue
            if self.adj[i][j] <= first:
                second = first
                first = self.adj[i][j]
            elif self.adj[i][j] <= second and self.adj[i][j] != first:
                second = self.adj[i][j]
        return second

    def tsp_rec(self, curr_bound, curr_weight, level, curr_path, visited):
        if level == self.N:
            if self.adj[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.adj[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.final_res:
                    self.copy_to_final(curr_path)
                    self.final_res = curr_res
            return

        for i in range(self.N):
            if (self.adj[curr_path[level - 1]][i] != 0 and not visited[i]):
                temp = curr_bound
                curr_weight += self.adj[curr_path[level - 1]][i]

                if level == 1:
                    curr_bound -= ((self.first_min(curr_path[level - 1]) + self.first_min(i)) / 2)
                else:
                    curr_bound -= ((self.second_min(curr_path[level - 1]) + self.first_min(i)) / 2)

                if curr_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    visited[i] = True
                    self.tsp_rec(curr_bound, curr_weight, level + 1, curr_path, visited)

                curr_weight -= self.adj[curr_path[level - 1]][i]
                curr_bound = temp
                visited = [False] * len(visited)
                for j in range(level):
                    if curr_path[j] != -1:
                        visited[curr_path[j]] = True

    def solve(self):
        curr_bound = 0
        curr_path = [-1] * (self.N + 1)
        visited = [False] * self.N

        for i in range(self.N):
            curr_bound += (self.first_min(i) + self.second_min(i))

        curr_bound = math.ceil(curr_bound / 2)
        visited[0] = True
        curr_path[0] = 0
        self.tsp_rec(curr_bound, 0, 1, curr_path, visited)

        return self.final_res, self.final_path

class Christofides:
    def __init__(self, graph):
        """
        Inicializa o algoritmo de Christofides para resolver o TSP.

        :param graph: Grafo do NetworkX representando o problema
        """
        self.graph = graph

    def solve(self):
        """
        Resolve o TSP utilizando o algoritmo de Christofides.

        :return: (caminho aproximado, custo do caminho)
        """
        # 1. Encontrar a árvore geradora mínima (MST)
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')

        # 2. Encontrar os nós de grau ímpar na MST
        odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]

        # Criar subgrafo completo dos nós de grau ímpar
        odd_graph = nx.Graph()
        for i in range(len(odd_degree_nodes)):
            for j in range(i + 1, len(odd_degree_nodes)):
                u, v = odd_degree_nodes[i], odd_degree_nodes[j]
                odd_graph.add_edge(u, v, weight=self.graph[u][v]['weight'])

        # Resolver o emparelhamento mínimo nos nós ímpares
        min_matching = nx.algorithms.matching.min_weight_matching(odd_graph, maxcardinality=True)

        # Adicionar as arestas do emparelhamento à MST
        multi_graph = nx.MultiGraph(mst)
        for u, v in min_matching:
            multi_graph.add_edge(u, v, weight=self.graph[u][v]['weight'])

        # 3. Encontrar um circuito Euleriano no grafo combinado
        eulerian_circuit = list(nx.eulerian_circuit(multi_graph))

        # Construir o ciclo Hamiltoniano eliminando nós duplicados
        visited = set()
        hamiltonian_path = []
        for u, _ in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                hamiltonian_path.append(u)
        hamiltonian_path.append(hamiltonian_path[0])  # Retorna ao início

        # Calcular o custo do ciclo Hamiltoniano
        cost = sum(self.graph[hamiltonian_path[i]][hamiltonian_path[i + 1]]['weight']
                   for i in range(len(hamiltonian_path) - 1))

        return hamiltonian_path, cost


class TwiceAroundTheTree:
    def __init__(self, graph):
        """
        Inicializa o algoritmo Twice-Around-the-Tree para resolver o TSP.

        :param graph: Grafo do NetworkX representando o problema
        """
        self.graph = graph

    def solve(self):
        """
        Resolve o TSP utilizando Twice-Around-the-Tree.

        :return: (caminho aproximado, custo do caminho)
        """
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')

        eulerian_graph = nx.MultiGraph(mst)
        eulerian_graph.add_edges_from(mst.edges(data=True))
        eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))

        visited = set()
        hamiltonian_path = []
        for u, v in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                hamiltonian_path.append(u)
        hamiltonian_path.append(hamiltonian_path[0])  # Retorna ao início

        cost = sum(self.graph[hamiltonian_path[i]][hamiltonian_path[i + 1]]['weight']
                   for i in range(len(hamiltonian_path) - 1))

        return hamiltonian_path, cost

import time
import tracemalloc
import os

# Função para processar um arquivo específico
def process_file(filepath):
    print(f"\nProcessando arquivo: {filepath}")
    # Carregar o grafo da instância TSPLIB
    graph = read_tsplib(filepath)
    print(f"Grafo carregado com sucesso do arquivo {filepath}!")

    # Resolver o problema do TSP com Twice-Around-the-Tree
    print("\nTwice-Around-the-Tree:")
    start_time_tat = time.time()
    start_mem_tat = tracemalloc.take_snapshot()
    solver_tat = TwiceAroundTheTree(graph)
    best_path_tat, best_cost_tat = solver_tat.solve()
    end_mem_tat = tracemalloc.take_snapshot()
    end_time_tat = time.time()
    mem_stats_tat = end_mem_tat.compare_to(start_mem_tat, "lineno")
    total_mem_tat = sum(stat.size_diff for stat in mem_stats_tat)
    print("Caminho aproximado:", best_path_tat)
    print("Custo do caminho:", best_cost_tat)
    print("Tempo de execução: {:.6f} segundos".format(end_time_tat - start_time_tat))
    print("Memória usada: {:.2f} KB".format(total_mem_tat / 1024))

    # Resolver o problema do TSP com Christofides
    print("\nChristofides:")
    start_time_christofides = time.time()
    start_mem_christofides = tracemalloc.take_snapshot()
    solver_christofides = Christofides(graph)
    best_path_christofides, best_cost_christofides = solver_christofides.solve()
    end_mem_christofides = tracemalloc.take_snapshot()
    end_time_christofides = time.time()
    mem_stats_christofides = end_mem_christofides.compare_to(start_mem_christofides, "lineno")
    total_mem_christofides = sum(stat.size_diff for stat in mem_stats_christofides)
    print("Caminho aproximado:", best_path_christofides)
    print("Custo do caminho:", best_cost_christofides)
    print("Tempo de execução: {:.6f} segundos".format(end_time_christofides - start_time_christofides))
    print("Memória usada: {:.2f} KB".format(total_mem_christofides / 1024))

    # Resolver o problema do TSP com Branch and Bound
    print("\nBranch and Bound:")
    adj_matrix = graph_to_adj_matrix(graph)
    start_time_bb = time.time()
    start_mem_bb = tracemalloc.take_snapshot()
    solver_bb = BranchAndBound(adj_matrix)
    best_cost_bb, best_path_bb = solver_bb.solve()
    end_mem_bb = tracemalloc.take_snapshot()
    end_time_bb = time.time()
    mem_stats_bb = end_mem_bb.compare_to(start_mem_bb, "lineno")
    total_mem_bb = sum(stat.size_diff for stat in mem_stats_bb)
    print("Melhor caminho:", best_path_bb)
    print("Custo do melhor caminho:", best_cost_bb)
    print("Tempo de execução: {:.6f} segundos".format(end_time_bb - start_time_bb))
    print("Memória usada: {:.2f} KB".format(total_mem_bb / 1024))


if __name__ == "__main__":
    # Diretório contendo os arquivos TSPLIB
    data_dir = "data"

    # Nomes dos arquivos na ordem desejada
    filenames = ["10.tsp", "20.tsp", "30.tsp", "40.tsp"]

    # Iniciar rastreamento de memória
    tracemalloc.start()

    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"Arquivo não encontrado: {filepath}")

    # Parar rastreamento de memória
    tracemalloc.stop()
