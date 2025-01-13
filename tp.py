import numpy as np
import networkx as nx
import math
import time
import tracemalloc
import os
import gc
from utils import read_tsplib

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
        Configura o algoritmo Branch and Bound para resolver o problema do TSP.

        :param adj_matrix: Matriz de adjacência representando o problema.
        """
        self.adj_matrix = adj_matrix
        self.num_nodes = len(adj_matrix)
        self.best_path = []
        self.min_cost = float('inf')

    def _calculate_min_edges(self, node):
        """
        Calcula o menor e o segundo menor peso de arestas conectadas a um nó.

        :param node: Índice do nó para cálculo.
        :return: Menor e segundo menor peso.
        """
        edges = [self.adj_matrix[node][i] for i in range(self.num_nodes) if i != node]
        first, second = sorted(edges)[:2] if len(edges) > 1 else (edges[0], float('inf'))
        return first, second

    def _explore(self, current_bound, current_cost, level, current_path, visited):
        """
        Realiza a recursão para explorar todas as possibilidades de caminhos.

        :param current_bound: Limite inferior para o caminho atual.
        :param current_cost: Custo acumulado do caminho atual.
        :param level: Nível da recursão.
        :param current_path: Caminho atual.
        :param visited: Conjunto de nós já visitados.
        """
        if level == self.num_nodes:
            final_cost = current_cost + self.adj_matrix[current_path[-1]][current_path[0]]
            if final_cost < self.min_cost:
                self.min_cost = final_cost
                self.best_path = current_path[:]
            return

        for i in range(self.num_nodes):
            if not visited[i] and self.adj_matrix[current_path[-1]][i] != float('inf'):
                temp_bound = current_bound
                temp_cost = current_cost + self.adj_matrix[current_path[-1]][i]

                if level == 1:
                    temp_bound -= self._calculate_min_edges(current_path[-1])[0] / 2
                else:
                    temp_bound -= sum(self._calculate_min_edges(current_path[-1])) / 2

                if temp_cost + temp_bound < self.min_cost:
                    visited[i] = True
                    current_path.append(i)

                    self._explore(temp_bound, temp_cost, level + 1, current_path, visited)

                    visited[i] = False
                    current_path.pop()

    def solve(self):
        """
        Resolve o problema do TSP com o algoritmo Branch and Bound.

        :return: (custo mínimo, melhor caminho)
        """
        current_bound = sum(sum(self._calculate_min_edges(i)) for i in range(self.num_nodes)) / 2
        visited = [False] * self.num_nodes
        visited[0] = True

        self._explore(current_bound, 0, 1, [0], visited)
        return self.min_cost, self.best_path

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
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')
        odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]

        odd_graph = nx.Graph()
        for i in range(len(odd_degree_nodes)):
            for j in range(i + 1, len(odd_degree_nodes)):
                u, v = odd_degree_nodes[i], odd_degree_nodes[j]
                odd_graph.add_edge(u, v, weight=self.graph[u][v]['weight'])

        min_matching = nx.algorithms.matching.min_weight_matching(odd_graph, maxcardinality=True)

        multi_graph = nx.MultiGraph(mst)
        for u, v in min_matching:
            multi_graph.add_edge(u, v, weight=self.graph[u][v]['weight'])

        eulerian_circuit = list(nx.eulerian_circuit(multi_graph))

        visited = set()
        hamiltonian_path = []
        for u, _ in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                hamiltonian_path.append(u)
        hamiltonian_path.append(hamiltonian_path[0])

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
        hamiltonian_path.append(hamiltonian_path[0])

        cost = sum(self.graph[hamiltonian_path[i]][hamiltonian_path[i + 1]]['weight']
                   for i in range(len(hamiltonian_path) - 1))

        return hamiltonian_path, cost

# Função para processar um arquivo específico
def process_file(filepath):
    print(f"\nProcessando arquivo: {filepath}")
    graph = read_tsplib(filepath)
    print(f"Grafo carregado com sucesso do arquivo {filepath}!")

    print("\nTwice-Around-the-Tree:")
    start_time_tat = time.time()
    tracemalloc.start()
    solver_tat = TwiceAroundTheTree(graph)
    best_path_tat, best_cost_tat = solver_tat.solve()
    end_time_tat = time.time()
    end_mem_tat = tracemalloc.take_snapshot()
    tracemalloc.stop()
    mem_stats_tat = end_mem_tat.statistics('lineno')
    total_mem_tat = sum(stat.size for stat in mem_stats_tat)

    print("Caminho aproximado:", best_path_tat)
    print("Custo do caminho:", best_cost_tat)
    print("Tempo de execução: {:.6f} segundos".format(end_time_tat - start_time_tat))
    print("Memória usada: {:.2f} KB".format(total_mem_tat / 1024))

    print("\nChristofides:")
    start_time_christofides = time.time()
    tracemalloc.start()
    solver_christofides = Christofides(graph)
    best_path_christofides, best_cost_christofides = solver_christofides.solve()
    end_time_christofides = time.time()
    end_mem_christofides = tracemalloc.take_snapshot()
    tracemalloc.stop()
    mem_stats_christofides = end_mem_christofides.statistics('lineno')
    total_mem_christofides = sum(stat.size for stat in mem_stats_christofides)

    print("Caminho aproximado:", best_path_christofides)
    print("Custo do caminho:", best_cost_christofides)
    print("Tempo de execução: {:.6f} segundos".format(end_time_christofides - start_time_christofides))
    print("Memória usada: {:.2f} KB".format(total_mem_christofides / 1024))

    print("\nBranch and Bound:")
    adj_matrix = graph_to_adj_matrix(graph)

    gc.collect()
    tracemalloc.start()
    start_time_bb = time.time()
    solver_bb = BranchAndBound(adj_matrix)
    best_cost_bb, best_path_bb = solver_bb.solve()
    end_time_bb = time.time()
    end_mem_bb = tracemalloc.take_snapshot()
    tracemalloc.stop()
    mem_stats_bb = end_mem_bb.statistics('lineno')
    total_mem_bb = sum(stat.size for stat in mem_stats_bb)

    print("Melhor caminho:", best_path_bb)
    print("Custo do melhor caminho:", best_cost_bb)
    print("Tempo de execução: {:.6f} segundos".format(end_time_bb - start_time_bb))
    print("Memória usada: {:.2f} KB".format(total_mem_bb / 1024))

if __name__ == "__main__":
    data_dir = "data"
    filenames = ["10.tsp", "20.tsp", "30.tsp", "40.tsp"]

    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"Arquivo não encontrado: {filepath}")