from gurobipy import *
import folium
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
from Visualization import visualize_routes, visualize_schedule_random, visualize_routes_static, visualize_routes_terminals
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
    def __init__(self, barge_id, capacity, fixed_cost,origin):
        self.id = barge_id                    # Unique identifier for the barge
        self.capacity = capacity              # Qk: Maximum capacity of the barge
        self.fixed_cost = fixed_cost    #Hk^B: Fixed cost associated with using the barge
        self.origin = origin

class Truck:
    def __init__(self, cost_per_container):
        self.cost_per_container = cost_per_container  # HT: Cost per container transported by truck

def check_model_status(model):
    """
    Checks the status of the Gurobi optimization model and handles various scenarios.
    Args:
        model (gurobipy.Model): The Gurobi model to check.
    """
    status = model.Status
    if status != GRB.OPTIMAL and status != GRB.INTERRUPTED:
        if status == GRB.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
        elif status == GRB.INFEASIBLE:
            print('The model is infeasible; computing IIS')
            model.computeIIS()
            print('The following constraint(s) cannot be satisfied:')
            for c in model.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)
        elif status != GRB.INF_OR_UNBD:
            print('Optimization was stopped with status', status)
        exit(0)

def random_generation_containers(nodes, node_coords, buffer_time=24*60):
    container_amount = 100  # Number of containers to generate
    containers_data = []

    n_depots = 5  # Node IDs 0-4
    n_terminals = 20  # Node IDs 5-24

    # Seed for reproducibility
    random.seed(42)

    for c in range(container_amount):
        container_id = c  # Unique container ID
        size = random.choice([1, 2])  # Size: 1 or 2
        container_type = random.choice(["E", "I"])

        if container_type == "E":
            # Opening date within first 24 hours
            opening_date = random.randint(0, 24*60)  # 0 to 1440 minutes
            # Closing date is at least buffer_time after opening_date
            max_closing_date = 24*60 + 172*60  # Up to ~3 days
            closing_date = opening_date + buffer_time + random.randint(0, max_closing_date - (opening_date + buffer_time))
            # Release date is before or at opening_date
            release_date = random.randint(0, opening_date)

            origin = random.randint(0, 4)  # Random depot origin (0-4)
            destination = random.randint(5, 24)  # Random terminal destination (5-24)

        else:  # Import containers
            release_date = None
            # Opening date within first 24 hours
            opening_date = random.randint(0, 24*60)  # 0 to 1440 minutes
            # Closing date is at least buffer_time after opening_date
            max_closing_date = 24*60 + 172*60  # Up to ~3 days
            closing_date = opening_date + buffer_time + random.randint(0, max_closing_date - (opening_date + buffer_time))

            origin = random.randint(5, 24)  # Random terminal origin (5-24)
            destination = random.randint(25, 29)  # Random depot arrival destination (25-29)

        # Cap closing_date to maximum allowed
        if closing_date > 196*60:
            closing_date = 196*60

        # Append the container data as a tuple
        containers_data.append(
            (
                container_id,
                size,
                release_date,
                opening_date,
                closing_date,
                origin,
                destination,
                container_type
            )
        )

    return containers_data  # Ensure the data is returned

from geopy.distance import geodesic
import random

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
        depot_to_dummy (dict): Dictionary mapping depot IDs to their dummy arrival node IDs.
    """
    # Define nodes with their types
    # Depots: 0-4 (updated to inland locations)
    # Depots' dummy arrival nodes: 25-29
    # Terminals: 5-24

    nodes = {
        0: Node(0, 'depot'),  # Veghel Depot
        1: Node(1, 'depot'),  #  (e.g., Tilburg Depot)
        2: Node(2, 'depot'),  # (e.g., Eindhoven Depot)
        3: Node(3, 'depot'),  #  (e.g., Nijmegen Depot)
        4: Node(4, 'depot'),  # (e.g., Utrecht Depot)

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

        25: Node(25, 'depot_arr'),  # Dummy arrival for Veghel Depot
        26: Node(26, 'depot_arr'),  # Tilburg
        27: Node(27, 'depot_arr'),  # Eindhoven
        28: Node(28, 'depot_arr'),  # Nijmegen
        29: Node(29, 'depot_arr')   # Utrecht
    }

    # Define coordinates for each node (latitude, longitude)
    # Updated depot coordinates to reflect inland locations like Veghel
    node_coords = {
        # Depots (Inland Locations)
        0: (51.5022, 5.6875),  # Veghel Depot
        1: (51.5667, 5.0689),  # Tilburg Depot
        2: (51.4416, 5.4697),  # Eindhoven Depot
        3: (51.8126, 5.8372),  # Nijmegen Depot
        4: (52.0907, 5.1214),  # Utrecht Depot

        # Terminals (Broader Rotterdam Area)
        5: (51.9200, 3.9900),  # RWG (West)
        6: (51.8800, 4.2500),  # Maasvlakte Terminal (Central West)
        7: (51.9450, 4.1000),  # ECT Delta (Northwest)
        8: (51.9100, 4.1500),  # ECT Euromax (Central)
        9: (51.8650, 4.3300),  # RST (Southeast)
        10: (51.9000, 4.2900),  # Uniport Multipurpose (South-Central)
        11: (51.8850, 4.3800),  # Steinweg Beatrixhaven (Far Southeast)
        12: (51.9400, 4.0600),  # RWG2 (Northwest)
        13: (51.8800, 4.2000),  # Miro Terminal (Southwest)
        14: (51.9150, 4.1300),  # Vopak (Central North)
        15: (51.9000, 4.2500),  # Stolthaven (Central South)
        16: (51.9350, 4.0200),  # Euro Tank Terminal (Far Northwest)
        17: (51.8600, 4.2700),  # Botlek Tank Terminal (Southeast)
        18: (51.9500, 4.1100),  # EMO (Central North)
        19: (51.9300, 3.9500),  # OBA Bulk Terminal (Far West)
        20: (51.8900, 4.3000),  # Gate LNG Terminal (Southeast)
        21: (51.9200, 4.2100),  # Odfjell Terminal (Central)
        22: (51.8500, 4.3700),  # Maasvlakte Olie Terminal (Far Southeast)
        23: (51.9400, 3.9700),  # Koole Tankstorage (Northwest)
        24: (51.8600, 4.1800),  # Botlek Terminal (Southwest)

        # Dummy arrival nodes for depots (same coords as their corresponding depot)
        25: (51.5022, 5.6875),  # Veghel Depot_arr
        26: (51.5667, 5.0689),  # Tilburg Depot_arr
        27: (51.4416, 5.4697),  # Eindhoven Depot_arr
        28: (51.8126, 5.8372),  # Nijmegen Depot_arr
        29: (52.0907, 5.1214)  # Utrecht Depot_arr
    }

    depot_to_dummy = {
        0: 25,  # depot 0 matches with dummy node 25
        1: 26,
        2: 27,
        3: 28,
        4: 29
    }

    # Define containers with their attributes
    containers_data = random_generation_containers(nodes, node_coords)

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
                travel_time = distance / 20 * 60  # Convert speed to travel time in minutes
                Tij[(i, j)] = travel_time

    # Create arcs based on calculated travel times
    arcs = []
    for (i, j), time in Tij.items():
        arc = Arc(i, j, time)  # Create Arc object
        arcs.append(arc)        # Add to arcs list
        nodes[i].add_out_arc(arc)  # Add to origin node's outgoing arcs
        nodes[j].add_in_arc(arc)   # Add to destination node's incoming arcs

    # Define barges with their capacities and fixed costs
    barges_data = [
        (1, 104, 3600, 0),  # Barge 1: Capacity=104, Fixed Cost=3600
        (2, 99, 3500, 1),
        (3, 81, 2800, 2),
        (4, 52, 1800, 3),
        (5, 28, 3700, 4)
    ]
    barges = {barge_id: Barge(barge_id, capacity, fixed_cost, origin)
              for barge_id, capacity, fixed_cost, origin in barges_data}

    # Define trucks with their cost per container
    HT = {1: 140,
          2: 200}  # You can add more truck IDs and their costs if needed

    truck = Truck(cost_per_container=HT)

    return nodes, arcs, containers, barges, truck, HT, node_coords, depot_to_dummy



def print_model_result(model, variables, barges, containers):
    """
    Prints the results of the optimization model, including objective value, container allocations, and barge routes.
    Args:
        model (gurobipy.Model): The optimized Gurobi model.
        variables (dict): Dictionary containing model variables and their values.
        barges (dict): Dictionary of Barge objects.
        containers (dict): Dictionary of Container objects.
    """
    print("\nOptimal Objective Value:", model.ObjVal)
    print("\nContainer Allocations:")
    f_ck = variables['f_ck']  # Container to vehicle allocation variables
    for c in containers.values():
        assigned = False
        for k in barges.keys():
            if f_ck[c.id, k].X > 0.5:
                print(f"Container {c.id} is allocated to Barge {k} to route {c.origin}-{c.destination}")
                assigned = True
        if f_ck[c.id, 'T'].X > 0.5:
            print(f"Container {c.id} is allocated to Truck to  {c.origin}-{c.destination}")
            assigned = True
        if not assigned:
            print(f"Container {c.id} is not assigned to any vehicle.")

    print("\nBarge Routes:")
    x_ijk = variables['x_ijk']  # Barge route selection variables
    for k in barges.keys():
        print(f"\nBarge {k} Route:")
        route = []
        for (i, j), var in x_ijk[k].items():
            if var.X > 0.5:
                route.append((i, j))
        if route:
            for arc in route:
                print(f"{arc[0]} -> {arc[1]}")
        else:
            print("No route for this barge.")

#=============================================================================================================================
#  Optimization of the Model using Gurobi
#=============================================================================================================================

def barge_scheduling_problem(nodes, arcs, containers, barges, truck, HT, node_coords,depot_to_dummy):
    """
    Optimizes barge and truck scheduling for transporting containers between depots and terminals.
    Args:
        nodes (dict): Dictionary of Node objects.
        arcs (list): List of Arc objects representing possible routes.
        containers (dict): Dictionary of Container objects.
        barges (dict): Dictionary of Barge objects.
        truck (Truck): Truck object with associated costs.
        HT (dict): Dictionary mapping truck IDs to cost per container.
        node_coords (dict): Dictionary mapping node IDs to their (latitude, longitude) coordinates.
    """
    # Initialize model
    model = Model("BargeScheduling")

    # Big M
    M = 3000 # A large constant used in Big M method for conditional constraints

    # Define sets
    N = list(nodes.keys())                         # Set of all node IDs
    C = list(containers.keys())                    # Set of all container IDs
    E = [c.id for c in containers.values() if c.type == 'E']  # Export containers
    I = [c.id for c in containers.values() if c.type == 'I']  # Import containers
    K = list(barges.keys()) + ['T']                # Set of barges and 'T' representing trucks
    KB = list(barges.keys())                       # Set of barges only

    # Define parameters
    Wc = {c.id: c.size for c in containers.values()}  # Wc: Container sizes
    Rc = {c.id: c.release_date for c in containers.values() if c.type == 'E'}  # Rc: Release dates for export containers
    Oc = {c.id: c.opening_date for c in containers.values()}  # Oc: Opening dates for all containers
    Dc = {c.id: c.closing_date for c in containers.values()}  # Dc: Closing dates for all containers

    # Zcj: Indicator if container c is associated with node j
    Zcj = {}
    for c in containers.values():
        for j in N:
            if c.origin == j:
                Zcj[c.id,j] = 1
            elif c.destination == j:
                Zcj[c.id,j] = 1
            else:
                Zcj[c.id,j] = 0

    HBk = {k: barges[k].fixed_cost for k in barges.keys()}  # HBk: Fixed costs for each barge
    Qk = {k: barges[k].capacity for k in barges.keys()}     # Qk: Capacities for each barge
    Or = {k: barges[k].origin for k in barges.keys()} #origin for each barge
    Tij = {(arc.origin, arc.destination): arc.travel_time for arc in arcs}  # Tij: Travel times between nodes

    L = 15     # Handling time per container in hours (e.g., loading/unloading time)
    gamma = 50 # Penalty cost for visiting sea terminals

    #=========================================================================================================================
    #  Define Decision Variables
    #=========================================================================================================================

    # f_ck: Binary variable indicating if container c is assigned to vehicle k
    f_ck = {}
    for c in C:
        for k in K:
            f_ck[c, k] = model.addVar(vtype=GRB.BINARY, name=f"f_{c}_{k}")

    # x_ijk: Binary variable indicating if barge k traverses arc (i, j)
    x_ijk = {}
    for k in KB:
        x_ijk[k] = {}
        for i in N:
            for j in N:
                if i != j and (i, j) in Tij:
                    x_ijk[k][(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")


    # p_jk: Continuous variable representing import quantities loaded by barge k at terminal j
    # d_jk: Continuous variable representing export quantities unloaded by barge k at terminal j
    p_jk = {}
    d_jk = {}
    for k in KB:
        for j in N:
            p_jk[j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"p_{j}_{k}")
            d_jk[j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"d_{j}_{k}")

    # y_ijk: Continuous variable for import containers on arc (i, j) by barge k
    # z_ijk: Continuous variable for export containers on arc (i, j) by barge k
    y_ijk = {}
    z_ijk = {}
    for k in KB:
        y_ijk[k] = {}
        z_ijk[k] = {}
        for i in N:
            for j in N:
                if i != j and (i, j) in Tij:
                    y_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}_{j}_{k}")
                    z_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}_{k}")

    # t_jk: Continuous variable representing the arrival time of barge k at node j
    t_jk = {}
    for k in KB:
        for j in N:
            t_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{j}_{k}")

    #=========================================================================================================================
    #  Define Objective Function
    #=========================================================================================================================

    """
    The objective is to minimize the total cost, which includes:
    - Truck transportation costs.
    - Barge fixed costs when departing from depots.
    - Barge travel times (assuming cost proportional to time).
    - Penalties for visiting sea terminals unnecessarily.
    """
    model.setObjective(
        quicksum(f_ck[c, 'T'] * HT[Wc[c]] for c in C) +  # Truck costs: Sum over all containers assigned to trucks

        quicksum(x_ijk[k][i,j] * HBk[k] for k in KB for j in N for i in N if nodes[j].type == 'terminal' and nodes[i].type =="depot")
        +  # Barge fixed costs: Applied only when departing from depot to a terminal
        quicksum(Tij[(i, j)] * x_ijk[k][(i,j)] for k in KB for j in N for i in N if i!=j)
        + # Barge travel time costs: Sum of travel times for all traversed arcs by barges
        quicksum(gamma * x_ijk[k][(i, j)] for k in KB for i in N for j in N if i!=j and nodes[i].type == "terminal"),  # Penalty for visiting sea terminals
        GRB.MINIMIZE)

    #=========================================================================================================================
    #  Define Constraints
    #=========================================================================================================================

    # f_ck: Binary variable indicating if container c is assigned to vehicle k
    f_ck = {}
    for c in C:
        for k in K:
            f_ck[c, k] = model.addVar(vtype=GRB.BINARY, name=f"f_{c}_{k}")

    # x_ijk: Binary variable indicating if barge k traverses arc (i, j)
    x_ijk = {}
    for k in KB:
        x_ijk[k] = {}
        for i in N:
            for j in N:
                if i != j and (i, j) in Tij:
                    x_ijk[k][(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    # p_jk: Continuous variable representing import quantities loaded by barge k at terminal j
    # d_jk: Continuous variable representing export quantities unloaded by barge k at terminal j
    p_jk = {}
    d_jk = {}
    for k in KB:
        for j in N:
            p_jk[j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"p_{j}_{k}")
            d_jk[j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"d_{j}_{k}")

    # y_ijk: Continuous variable for import containers on arc (i, j) by barge k
    # z_ijk: Continuous variable for export containers on arc (i, j) by barge k
    y_ijk = {}
    z_ijk = {}
    for k in KB:
        y_ijk[k] = {}
        z_ijk[k] = {}
        for i in N:
            for j in N:
                if i != j and (i, j) in Tij:
                    y_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}_{j}_{k}")
                    z_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}_{k}")

    # t_jk: Continuous variable representing the arrival time of barge k at node j
    t_jk = {}
    for k in KB:
        for j in N:
            t_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{j}_{k}")

    # =========================================================================================================================
    #  Define Objective Function
    # =========================================================================================================================

    """
    The objective is to minimize the total cost, which includes:
    - Truck transportation costs.
    - Barge fixed costs when departing from depots.
    - Barge travel times (assuming cost proportional to time).
    - Penalties for visiting sea terminals unnecessarily.
    """
    model.setObjective(
        quicksum(f_ck[c, 'T'] * HT[Wc[c]] for c in C) +  # Truck costs: Sum over all containers assigned to trucks

        quicksum(x_ijk[k][i, j] * HBk[k] for k in KB for j in N for i in N if
                 nodes[j].type == 'terminal' and nodes[i].type == "depot")
        +  # Barge fixed costs: Applied only when departing from depots
        quicksum(Tij[(i, j)] * x_ijk[k][(i, j)] for k in KB for j in N for i in N if i != j)
        +  # Barge travel time costs: Sum of travel times for all traversed arcs by barges
        quicksum(gamma * x_ijk[k][(i, j)] for k in KB for i in N for j in N if i != j and nodes[i].type == "terminal"),
        # Penalty for visiting sea terminals
        GRB.MINIMIZE)

    # =========================================================================================================================
    #  Define Constraints
    # =========================================================================================================================

    # (1) Each container is allocated to exactly one barge or truck
    for c in C:
        model.addConstr(
            quicksum(f_ck[c, k] for k in K) == 1,
            name=f"Assignment_{c}"
        )
        # Explanation:
        # Ensures that each container is assigned to one and only one vehicle (either a barge or a truck)

    # CHANGED (2) Flow conservation for x_ijk (Barge Routes)
    for k in KB:
        origin_node = Or[k]  # Get the origin node for barge k
        destination_node = depot_to_dummy[origin_node]  # Map to the corresponding depot_arr node
        for i in N:
            if i == origin_node:
                # Flow conservation for the origin node of barge k
                model.addConstr(
                    (quicksum(x_ijk[k][(i, j)] for j in N if j != i and (i, j) in Tij) -
                     quicksum(x_ijk[k][(j, i)] for j in N if j != i and (j, i) in Tij))
                    == 1,
                    name=f"Flow_conservation_origin_{k}_{i}"
                )
            elif i == destination_node:
                # Flow conservation for the destination node of barge k
                model.addConstr(
                    (quicksum(x_ijk[k][(i, j)] for j in N if j != i and (i, j) in Tij) -
                     quicksum(x_ijk[k][(j, i)] for j in N if j != i and (j, i) in Tij))
                    == -1,
                    name=f"Flow_conservation_destination_{k}_{i}"
                )
            else:
                # Flow conservation for all other nodes
                model.addConstr(
                    (quicksum(x_ijk[k][(i, j)] for j in N if j != i and (i, j) in Tij) -
                     quicksum(x_ijk[k][(j, i)] for j in N if j != i and (j, i) in Tij))
                    == 0,
                    name=f"Flow_conservation_internal_{k}_{i}"
                )

    # (3) each barge is used at most once
    for k in KB:
        model.addConstr(
            quicksum(x_ijk[k][(i, j)] for j in N for i in N if
                     nodes[i].type == "depot" and i != j) <= 1,
            name=f"Barge_used_{k}"
        )

    # ADDED (31) Add constraints to ensure barges only carry containers from their origin depot
    for k in KB:
        origin_node = Or[k]
        for c in C:
            if containers[c].origin != origin_node and containers[c].type == 'E':
                model.addConstr(f_ck[c, k] == 0, name=f"Origin_constraint_{c}_{k}")

    # ADDED (32) add contraints to ensure barge visits destination node of container
    for c in E + I:
        destination = containers[c].destination
        for k in KB:
            # Barge k must enter the destination node if it carries container c
            model.addConstr(
                quicksum(x_ijk[k][(i, destination)] for i in N if (i, destination) in Tij) >= f_ck[c, k],
                name=f"Barge_{k}_traverse_destination_{c}"
            )

    # (4) Import quantities loaded by barge k at sea terminal j
    for k in KB:
        for j in N:
            if nodes[j].type == "terminal":
                model.addConstr(p_jk[j, k] == quicksum(Wc[c] * Zcj[c, j] * f_ck[c, k] for c in I),
                                name=f"import_quantities_{j}_{k}")

    # (5) Export quantitites loadded by barge k at sea termina j
    for k in KB:
        for j in N:
            if nodes[j].type == "terminal":
                model.addConstr(d_jk[j, k] == quicksum(Wc[c] * Zcj[c, j] * f_ck[c, k] for c in E),
                                name=f"Export_quantities_{j}_{k}")

    # (6) Flow equations for y_ijk (import containers)
    for k in KB:
        for j in N:
            if nodes[j].type == 'terminal':
                inflow = quicksum(y_ijk[k][(j, i)] for i in N if i != j)
                outflow = quicksum(y_ijk[k][(i, j)] for i in N if i != j)
                model.addConstr(inflow - outflow == p_jk[j, k], name=f"ImportFlow_{j}_{k}")
                # Explanation:
                # Ensures that the net inflow of import containers at terminal j by barge k equals the total imports loaded

    # (7) Flow equations for z_ijk (export containers)
    for k in KB:
        for j in N:
            if nodes[j].type == 'terminal':
                inflow = quicksum(z_ijk[k][(i, j)] for i in N if i != j)
                outflow = quicksum(z_ijk[k][(j, i)] for i in N if i != j)
                model.addConstr(
                    inflow - outflow == d_jk[j, k],
                    name=f"ExportFlow_{j}_{k}"
                )
                # Explanation:
                # Ensures that the net inflow of export containers at terminal j by barge k equals the total exports unloaded

    # (8)
    for k in KB:
        for i in N:
            for j in N:
                if i != j:
                    model.addConstr(y_ijk[k][(i, j)] + z_ijk[k][(i, j)] <= Qk[k] * x_ijk[k][(i, j)],
                                    name=f"Capacity_{i}_{j}_{k}")

    # (9) Barge departure time after release of export containers
    for c in E:
        for k in KB:
            if c in Rc:
                depot = containers[c].origin
                model.addConstr(
                    t_jk[depot, k] >= Rc[c] * f_ck[c, k],
                    name=f"BargeDeparture_{c}_{k}"
                )
                # Explanation:
                # If container c is assigned to barge k, ensure that barge k departs from the depot no earlier than the container's release date Rc[c]
                # If f_ck[c, k] = 0, the constraint becomes t_jk >= 0, which is always true

    #
    # (10)
    for k in KB:
        for i in N:
            for j in N:
                if i != j:
                    model.addConstr(
                        t_jk[j, k] >= t_jk[i, k] + quicksum(L * Zcj[c, i] * f_ck[c, k] for c in C) + Tij[(i, j)] - (
                                    1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeLB_{i}_{j}_{k}"
                    )

    # (11)
    for k in KB:
        for i in N:
            for j in N:
                if i != j:
                    model.addConstr(
                        t_jk[j, k] <= t_jk[i, k] + quicksum(L * Zcj[c, i] * f_ck[c, k] for c in C) + Tij[(i, j)] + (
                                    1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeUB_{i}_{j}_{k}"
                    )
    # (12)
    for c in C:
        for j in N:
            if nodes[j].type == 'terminal':  # Exclude depot
                for k in KB:
                    model.addConstr(
                        t_jk[j, k] >= Oc[c] * Zcj[c, j] - (1 - f_ck[c, k]) * M,
                        name=f"ReleaseTime_{c}_{j}_{k}"
                    )
    # (13)
    for c in C:
        for j in N:
            if nodes[j].type == 'terminal':  # Exclude depot
                for k in KB:
                    model.addConstr(
                        t_jk[j, k] * Zcj[c, j] <= Dc[c] + (1 - f_ck[c, k]) * M,
                        name=f"ClosingTime_{c}_{j}_{k}"
                    )
    # ADDED (14) time of delivery is after time of pickup
    for c in C:
        origin = containers[c].origin
        destination = containers[c].destination
        for k in KB:
            # Only apply if container c is assigned to barge k
            model.addConstr(
                t_jk[destination, k] >= t_jk[origin, k] - (1 - f_ck[c, k]) * M,
                name=f"Sequence_origin_before_destination_indirect_{k}_{c}"
            )

    #=========================================================================================================================
    #  Optimize the Model
    #=========================================================================================================================

    # Update the model with all variables and constraints
    model.update()

    # Set Gurobi parameters
    model.setParam('OutputFlag', True)    # Enable solver output
    model.setParam('TimeLimit', 1800)      # Set a time limit of 5 minutes (300 seconds)

    # Start the optimization process
    model.optimize()

    # Check the status of the model to ensure feasibility and optimality
    # check_model_status(model)

    #=========================================================================================================================
    #  Extract Variable Values
    #=========================================================================================================================

    # Extract values for f_ck variables (container allocations)
    f_ck_values = {}
    for key, var in f_ck.items():
        f_ck_values[key] = var  # Store the Gurobi variable object for later access (e.g., var.X)

    # Extract values for x_ijk variables (barge route selections)
    x_ijk_values = {}
    for k in KB:
        x_ijk_values[k] = {}
        for key, var in x_ijk[k].items():
            x_ijk_values[k][key] = var  # Store the Gurobi variable object

    # Extract values for t_jk variables (arrival times at nodes)
    t_jk_values = {}
    for key, var in t_jk.items():
        t_jk_values[key] = var.X  # Store the optimized value

    # Collect variables into a dictionary for ease of access
    variables = {
        'f_ck': f_ck_values,
        'x_ijk': x_ijk_values,
        't_jk': t_jk_values
    }

    #=========================================================================================================================
    #  Output Results and Visualization
    #=========================================================================================================================

    # Print the optimization results: objective value, container allocations, and barge routes
    print_model_result(model, variables, barges, containers)

    # Visualize the barge and truck routes on a map
    visualize_routes_static(nodes, barges, variables, containers, node_coords)
    visualize_routes(nodes, barges, variables, containers, node_coords,"folium_map_random.html")
    #
    # # Visualize the schedule in gantt chart format of container movements
    visualize_schedule_random(nodes, barges, variables, containers)
    visualize_routes_terminals(nodes, barges, variables, containers, node_coords, "barge_routes.png")






if __name__ == "__main__":
    nodes, arcs, containers, barges, truck, HT, node_coords,depot_to_dummy = construct_network()
    barge_scheduling_problem(nodes, arcs, containers, barges, truck, HT, node_coords,depot_to_dummy)
