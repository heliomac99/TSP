import numpy as np

def generate_tsp_instance(num_points, file_name):
    """
    Generate a TSP instance with the given number of points, ensuring the triangle inequality holds.
    :param num_points: Number of points in the instance.
    :param file_name: Name of the file to save the instance.
    """
    # Generate random 2D coordinates for the points
    points = np.random.rand(num_points, 2) * 100  # Scale to a 100x100 grid

    with open(file_name, 'w') as f:
        f.write("NAME : GeneratedInstance\n")
        f.write(f"TYPE : TSP\n")
        f.write(f"DIMENSION : {num_points}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")

        for i, (x, y) in enumerate(points, start=1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")

        f.write("EOF\n")

    print(f"Instance saved to {file_name}")

# Generate TSP instances of sizes 20, 30, and 40
generate_tsp_instance(20, "20.tsp")
generate_tsp_instance(30, "30.tsp")
generate_tsp_instance(40, "40.tsp")