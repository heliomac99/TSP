import argparse
import json
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


class BranchAndBound:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.num_nodes = len(adj_matrix)
        self.best_path = []
        self.min_cost = float('inf')

    def _calculate_min_edges(self, node):
        edges = [self.adj_matrix[node][i] for i in range(self.num_nodes) if i != node]
        first, second = sorted(edges)[:2] if len(edges) > 1 else (edges[0], float('inf'))
        return first, second

    def _explore(self, current_bound, current_cost, level, current_path, visited):
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
        current_bound = sum(sum(self._calculate_min_edges(i)) for i in range(self.num_nodes)) / 2
        visited = [False] * self.num_nodes
        visited[0] = True

        self._explore(current_bound, 0, 1, [0], visited)
        return self.best_path, self.min_cost


class Christofides:
    def __init__(self, graph):
        self.graph = graph

    def solve(self):
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')
        odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]

        odd_graph = nx.Graph()
        for i in range(len(odd_degree_nodes)):
            for j in range(i + 1, len(odd_degree_nodes)):
                u, v = odd_degree_nodes[i], odd_degree_nodes[j]
                odd_graph.add_edge(u, v, weight=self.graph[u][v]['weight'])

        min_matching = nx.algorithms.matching.min_weight_matching(odd_graph)

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
        self.graph = graph

    def solve(self):
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
    

def measure_performance(solver_class, graph, is_branch_and_bound=False):
    start_time = time.time()
    tracemalloc.start()

    if is_branch_and_bound:
        adj_matrix = graph_to_adj_matrix(graph)
        gc.collect()
        solver = solver_class(adj_matrix)
    else:
        solver = solver_class(graph)

    best_path, cost = solver.solve()
    end_time = time.time()

    end_mem = tracemalloc.take_snapshot()
    tracemalloc.stop()
    mem_stats = end_mem.statistics('lineno')
    total_mem = sum(stat.size for stat in mem_stats)

    return {
        "tempo": end_time - start_time,
        "memoria": total_mem / 1024,
        "tamanho": len(graph.nodes),
        "custo": cost
    }

def process_file(filepath, algorithm, opt=None):
    graph = read_tsplib(filepath)

    solver_map = {
        "twice": (TwiceAroundTheTree, False),
        "christofides": (Christofides, False),
        "branch": (BranchAndBound, True)
    }

    if algorithm not in solver_map:
        return None

    solver_class, is_branch_and_bound = solver_map[algorithm]
    performance_data = measure_performance(solver_class, graph, is_branch_and_bound)

    if opt is not None:
        performance_data["proporcao"] = float(performance_data["custo"]) / opt
        del performance_data["custo"]

    performance_data["tipo"] = algorithm
    return performance_data

if __name__ == "__main__":
    data_dir = "data"
    small_tests = ["5.tsp", "10.tsp", "15.tsp", "20.tsp"]
    large_tests = [
        ("berlin52.tsp", 7542),
        ("ch130.tsp", 6110),
        ("a280.tsp", 2579),
        ("d493.tsp", 35002),
        ("d657.tsp", 48912),
        ("rat783.tsp", 8806),
        ("u1060.tsp", 224094),
        ("d1291.tsp", 50801),
        ("u1817.tsp", 57201),
        ("u2319.tsp", 234256)
    ]

    results = []

    # Processar pequenos testes
    for filename in small_tests:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            branch_result = process_file(filepath, "branch")
            opt = float(branch_result["custo"])
            branch_result["proporcao"] = 1.0  # Branch retorna o ótimo
            del branch_result["custo"]
            results.append(branch_result)

            for algo in ["christofides", "twice"]:
                result = process_file(filepath, algo, opt=opt)
                results.append(result)
        else:
            print(f"Arquivo não encontrado: {filepath}")

    # Processar grandes testes
    for filename, opt in large_tests:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            for algo in ["christofides", "twice"]:
                result = process_file(filepath, algo, opt=opt)
                results.append(result)
        else:
            print(f"Arquivo não encontrado: {filepath}")

    # Salvar os resultados em um arquivo JSON na pasta 'results'
    os.makedirs('results', exist_ok=True)
    output_filename = os.path.join('results', "report.json")
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em '{output_filename}'")
