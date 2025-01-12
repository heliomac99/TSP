import numpy as np
import networkx as nx

def read_tsplib(file_path):
    """
    Lê um arquivo TSPLIB e cria um grafo do NetworkX.
    :param file_path: Caminho para o arquivo TSPLIB
    :return: Grafo do NetworkX
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    nodes = []
    start = False

    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            start = True
            continue
        if start:
            if line.strip() == "EOF":
                break
            parts = line.split()
            nodes.append((float(parts[1]), float(parts[2])))

    # Criar um grafo completo com as distâncias euclidianas
    graph = nx.Graph()
    for i, (x1, y1) in enumerate(nodes):
        for j, (x2, y2) in enumerate(nodes):
            if i != j:
                weight = np.hypot(x2 - x1, y2 - y1)
                graph.add_edge(i, j, weight=weight)

    return graph

if __name__ == "__main__":
    distance_matrix = read_tsplib("data/pequeno.tsp")
    print("Matriz de distâncias carregada com tamanho:", distance_matrix.shape)