from gurobipy import *
import folium
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
from Visualization import visualize_routes, visualize_schedule
import random

class Node:
    def __init__(self, node_id, node_type='terminal'):
        self.id = node_id
        self.type = node_type  # 'depot' or 'terminal'
        self.in_arcs = []       # List to store incoming arcs
        self.out_arcs = []      # List to store outgoing arcs

    def add_in_arc(self, arc):
        self.in_arcs.append(arc)

    def add_out_arc(self, arc):
        self.out_arcs.append(arc)

class Container:
    def __init__(self, container_id, size, release_date, opening_date, closing_date, origin, destination, container_type):
        self.id = container_id
        self.size = size                      # Wc: Size or weight of the container
        self.release_date = release_date      # Rc: Earliest time container is available for loading
        self.opening_date = opening_date      # Oc: Earliest time container can be unloaded at destination
        self.closing_date = closing_date      # Dc: Latest time container must be unloaded
        self.origin = origin                  # Origin node ID (depot or terminal)
        self.destination = destination        # Destination node ID (depot or terminal)
        self.type = container_type            # 'I' for import, 'E' for export

class Arc:
    def __init__(self, origin, destination, travel_time):
        self.origin = origin                  # Origin node ID
        self.destination = destination        # Destination node ID
        self.travel_time = travel_time        # Tij: Travel time in minutes between origin and destination

class Barge:
    def __init__(self, barge_id, capacity, fixed_cost):
        self.id = barge_id                    # Unique identifier for the barge
        self.capacity = capacity              # Qk: Maximum capacity of the barge
        self.fixed_cost = fixed_cost          # Hk^B: Fixed cost associated with using the barge

class Truck:
    def __init__(self, cost_per_container):
        self.cost_per_container = cost_per_container  # HT: Cost per container transported by truck

def random_generation_containers(nodes,node_coords):
    container_amount = 300  # Number of containers to generate
    containers_data = []

    n_depots = 5 #(0-4)
    n_terminals = 20 #(5-24)

    for c in range(container_amount):
        container_id = c  # Unique container ID
        size = random.choice([1, 2])  # Size: 1 or 2
        container_type = random.choice(["Export", "Import"])

        if container_type == "Export":
            release_date = random.randint(0, 24*60)
            closing_date = random.randint(25*60, 196*60)
            opening_date = random.randint(0, closing_date - 24*60)

            #create list of all "depot" nodes to choose from

            origin = random.randint(0, 4)  # Random origin
            destination = random.randint(5, 24)  # Random destination

        else:  # if container_type == "Import":
            release_date = 0
            closing_date = random.randint(25*60, 196 * 60)
            opening_date = random.randint(0, closing_date - 24 * 60)

            destination = random.randint(25, 29)  # Random depot destination
            origin = random.randint(5, 24)  # Random terminal origin

        # Add the tuple to the list
        containers_data.append(
            (container_id, size, release_date, opening_date, closing_date, origin, destination, container_type))

    # Initialize containers dictionary



    return containers_data

def construct_network():
    """
    Constructs the transportation network by defining nodes, containers, arcs, barges, and trucks.
    Returns:
        nodes (dict): Dictionary of Node objects keyed by node ID.
        arcs (list): List of Arc objects representing possible routes.
        containers (dict): Dictionary of Container objects keyed by container ID.
        barges (dict): Dictionary of Barge objects keyed by barge ID.
        truck (Truck): Truck object with associated costs.
        HT (dict): Dictionary mapping truck IDs to their cost per container.
        node_coords (dict): Dictionary mapping node IDs to their (latitude, longitude) coordinates.
    """
    # Define nodes with their types
    # Define nodes with their types
    # Depots: 0-4
    # Depots' dummy arrival nodes: 25-29
    # Terminals: 5-24

    nodes = {
        0: Node(0, 'depot'),  # Eemhaven Depot
        1: Node(1, 'depot'),  # Waalhaven Depot
        2: Node(2, 'depot'),  # Pernis Depot
        3: Node(3, 'depot'),  # Botlek Depot
        4: Node(4, 'depot'),  # Europoort Depot

        5: Node(5, 'terminal'),  # RWG
        6: Node(6, 'terminal'),  # APMT Maasvlakte II
        7: Node(7, 'terminal'),  # ECT Delta
        8: Node(8, 'terminal'),  # ECT Euromax
        9: Node(9, 'terminal'),  # APMT (older)
        10: Node(10, 'terminal'),  # Uniport Multipurpose
        11: Node(11, 'terminal'),  # RST (Rotterdam Shortsea Terminals)
        12: Node(12, 'terminal'),  # Steinweg Beatrixhaven
        13: Node(13, 'terminal'),  # RWG2
        14: Node(14, 'terminal'),  # Miro Terminal
        15: Node(15, 'terminal'),  # Vopak
        16: Node(16, 'terminal'),  # Stolthaven
        17: Node(17, 'terminal'),  # Euro Tank Terminal (ETT)
        18: Node(18, 'terminal'),  # Botlek Tank Terminal
        19: Node(19, 'terminal'),  # OBA Bulk Terminal
        20: Node(20, 'terminal'),  # EMO
        21: Node(21, 'terminal'),  # Gate LNG Terminal
        22: Node(22, 'terminal'),  # Odfjell Terminal
        23: Node(23, 'terminal'),  # Maasvlakte Olie Terminal
        24: Node(24, 'terminal'),  # Koole Tankstorage

        25: Node(25, 'depot_arr'),  # Dummy arrival for Eemhaven Depot
        26: Node(26, 'depot_arr'),  # Dummy arrival for Waalhaven Depot
        27: Node(27, 'depot_arr'),  # Dummy arrival for Pernis Depot
        28: Node(28, 'depot_arr'),  # Dummy arrival for Botlek Depot
        29: Node(29, 'depot_arr')  # Dummy arrival for Europoort Depot
    }

    # Define coordinates for each node (latitude, longitude)
    # Coordinates are approximations around the Port of Rotterdam area.
    node_coords = {
        # Depots
        0: (51.9000, 4.4500),  # Eemhaven Depot
        1: (51.8950, 4.4100),  # Waalhaven Depot
        2: (51.8850, 4.3700),  # Pernis Depot
        3: (51.8900, 4.3000),  # Botlek Depot
        4: (51.9550, 4.1800),  # Europoort Depot

        # Terminals (scattered throughout the port area)
        5: (51.9495, 4.0290),  # RWG
        6: (51.9530, 3.9900),  # APMT Maasvlakte II
        7: (51.9520, 4.0000),  # ECT Delta
        8: (51.9555, 4.0050),  # ECT Euromax
        9: (51.9570, 4.0100),  # APMT (older)
        10: (51.9150, 4.4200),  # Uniport Multipurpose
        11: (51.9100, 4.4000),  # RST (Rotterdam Shortsea Terminals)
        12: (51.9050, 4.4900),  # Steinweg Beatrixhaven
        13: (51.9500, 4.0150),  # RWG2
        14: (51.9105, 4.3200),  # Miro
        15: (51.9200, 4.3400),  # Vopak
        16: (51.9250, 4.3800),  # Stolthaven
        17: (51.9350, 4.3200),  # Euro Tank Terminal (ETT)
        18: (51.8955, 4.2950),  # Botlek Tank Terminal
        19: (51.9700, 4.1500),  # OBA Bulk Terminal
        20: (51.9600, 4.1200),  # EMO
        21: (51.9630, 4.0500),  # Gate LNG Terminal
        22: (51.8800, 4.3850),  # Odfjell Terminal
        23: (51.9505, 3.9700),  # Maasvlakte Olie Terminal
        24: (51.9070, 4.4400),  # Koole Tankstorage

        # Dummy arrival nodes for depots (same coords as their corresponding depot)
        25: (51.9000, 4.4500),  # Eemhaven Depot_arr
        26: (51.8950, 4.4100),  # Waalhaven Depot_arr
        27: (51.8850, 4.3700),  # Pernis Depot_arr
        28: (51.8900, 4.3000),  # Botlek Depot_arr
        29: (51.9550, 4.1800)  # Europoort Depot_arr
    }

    # Define containers with their attributes
    containers_data = random_generation_containers(nodes,node_coords)

    # Initialize containers dictionary
    containers = {}
    for data in containers_data:
        c = Container(*data)      # Unpack data into Container constructor
        containers[c.id] = c      # Add to containers dictionary

    # Calculate travel times between nodes (in minutes)
    # Assume average speed of 20 km/h for barges and trucks
    Tij = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                distance = geodesic(node_coords[i], node_coords[j]).kilometers  # Calculate distance in km
                travel_time = distance / 20 * 60 # Convert speed to travel time in hours
                Tij[(i, j)] = travel_time

    # Create arcs based on calculated travel times
    arcs = []
    for (i, j), time in Tij.items():
        arc = Arc(i, j, time)       # Create Arc object
        arcs.append(arc)            # Add to arcs list
        nodes[i].add_out_arc(arc)   # Add to origin node's outgoing arcs
        nodes[j].add_in_arc(arc)    # Add to destination node's incoming arcs

    # Define barges with their capacities and fixed costs
    barges_data = [
        (1, 104, 3600),  # Barge 1: Capacity=104, Fixed Cost=3600
        (2, 99, 3500),
        (2, 81, 2800),
        (2, 52, 1800),
        (2, 28, 3700) #
    ]
    barges = {barge_id: Barge(barge_id, capacity, fixed_cost)
              for barge_id, capacity, fixed_cost in barges_data}

    # Define trucks with their cost per container
    HT = {1: 200}
                             # change time per truck also if you want to change the cost

    truck = Truck(cost_per_container=HT)
    return nodes, arcs, containers, barges, truck, HT, node_coords



if __name__ == "__main__":
    nodes, arcs, containers, barges, truck, HT, node_coords = construct_network()
    print(len(containers))