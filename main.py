from gurobipy import *
import folium
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
from Visualization import visualize_routes, visualize_schedule

#=============================================================================================================================
#  Define Classes for Nodes, Containers, Arcs, Barges, and Trucks
#=============================================================================================================================

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

#=============================================================================================================================
#  Auxiliary Functions
#=============================================================================================================================

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
    nodes = {
        0: Node(0, 'depot'),  # Depot 1
        1: Node(1, 'depot'),  # Depot 2
        2: Node(2, 'terminal'),  # Terminal 1
        3: Node(3, 'terminal'),  # Terminal 2
        4: Node(4, 'terminal'),  # Terminal 3
        5: Node(5, 'terminal'),  # Terminal 4
        6: Node(6, 'terminal'),  # Terminal 5
        7: Node(7, "depot_arr"), # dumy depot 1
        8: Node(8, "depot_arr")  # dumy depot 2
    }

    # Define coordinates for each node (latitude, longitude)
    node_coords = {
        0: (51.957979, 4.052421),  # Depot 0
        1: (51.912345, 4.234567),  # Depot 1
        2: (51.948060, 4.063992),  # Terminal 1
        3: (51.955480, 4.052490),  # Terminal 2
        4: (51.961111, 4.034722),  # Terminal 3
        5: (51.906540, 4.122230),  # Terminal 4
        6: (51.904251, 4.141221),
        7: (51.957979, 4.052421),  # Depot 0
        8: (51.912345, 4.234567)
        # Terminal 5
    }

    depot_to_dummy = {
        0: 7,  # depot 0 should match with dummy node 7
        1: 8  # depot 1 should match with dummy node 8

    }

    # Define containers with their attributes
    containers_data = [
        # container_id, size, release_date, opening_date, closing_date, origin, destination, container_type
        (1, 1, 2, 5, 12, 0, 2, 'E'),  # From Depot 0 to Terminal 1
        (2, 2, 4, 6, 10, 1, 3, 'E'),  # From Depot 1 to Terminal 2
        (3, 1, 1, 4, 8, 0, 4, 'E'),  # From Depot 0 to Terminal 3
        (4, 2, 3, 7, 9, 1, 5, 'E'),  # From Depot 1 to Terminal 4
        (5, 1, 5, 9, 13, 0, 6, 'E'),  # From Depot 0 to Terminal 5
        (6, 2, 6, 10, 14, 1, 2, 'E'),  # From Depot 1 to Terminal 1
        # Import Containers (from Terminals to Depots)
        (7, 1, None, 3, 9, 2, 7, 'I'),  # From Terminal 1 to Depot 0
        (8, 2, None, 2, 7, 3, 8, 'I'),  # From Terminal 2 to Depot 1
        (9, 1, None, 5, 11, 4, 7, 'I'),  # From Terminal 3 to Depot 0
        (10, 2, None, 6, 12, 5, 8, 'I'),  # From Terminal 4 to Depot 1
        (11, 1, None, 8, 14, 6, 7, 'I'),  # From Terminal 5 to Depot 0
        (12, 2, None, 9, 15, 2, 8, 'I'),  # From Terminal 1 to Depot 1
    ]


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
                travel_time = distance / 20  # Convert speed to travel time in hours
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
        (1, 8, 1000,0),  # Barge 1: Capacity=10, Fixed Cost=1000, origin

        (2, 8, 1000,1),  # Barge 2: Capacity=10, Fixed Cost=1000, origin
        (3,5,1000,0)
    ]
    barges = {barge_id: Barge(barge_id, capacity, fixed_cost, origin)
              for barge_id, capacity, fixed_cost,origin in barges_data}

    # Define trucks with their cost per container
    HT = {1: 500,
          2:1000}
                             # change time per truck also if you want to change the cost

    truck = Truck(cost_per_container=HT)
    return nodes, arcs, containers, barges, truck, HT, node_coords,depot_to_dummy

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
    M =2000  # A large constant used in Big M method for conditional constraints

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

    L = 0.05      # Handling time per container in hours (e.g., loading/unloading time)
    gamma = 0.05 # Penalty cost for visiting sea terminals

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
        +  # Barge fixed costs: Applied only when departing from depots
        quicksum(Tij[(i, j)] * x_ijk[k][(i,j)] for k in KB for j in N for i in N if i!=j)
        + # Barge travel time costs: Sum of travel times for all traversed arcs by barges
        quicksum(gamma * x_ijk[k][(i, j)] for k in KB for i in N for j in N if i!=j and nodes[i].type == "terminal"),  # Penalty for visiting sea terminals
        GRB.MINIMIZE)

    #=========================================================================================================================
    #  Define Constraints
    #=========================================================================================================================

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
            quicksum(x_ijk[k][(i,j)] for j in N for i in N if
                     nodes[i].type == "depot" and i!=j) <= 1,
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
                model.addConstr(p_jk[j,k] == quicksum(Wc[c] * Zcj[c, j] * f_ck[c, k] for c in I),
                                name = f"import_quantities_{j}_{k}")

    # (5) Export quantitites loadded by barge k at sea termina j
    for k in KB:
        for j in N:
            if nodes[j].type == "terminal":
                model.addConstr(d_jk[j,k] == quicksum(Wc[c] * Zcj[c, j] * f_ck[c, k] for c in E),
                                name = f"Export_quantities_{j}_{k}")

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
                if i!=j:
                    model.addConstr(y_ijk[k][(i,j)] + z_ijk[k][(i,j)] <= Qk[k] * x_ijk[k][(i,j)], name=f"Capacity_{i}_{j}_{k}")

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
                        t_jk[j, k] >= t_jk[i, k] + quicksum(L * Zcj[c, i] * f_ck[c, k] for c in C) + Tij[(i, j)] - (1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeLB_{i}_{j}_{k}"
                    )

    #(11)
    for k in KB:
        for i in N:
            for j in N:
                if i != j:
                    model.addConstr(
                                t_jk[j, k] <= t_jk[i, k] + quicksum(L * Zcj[c, i] * f_ck[c, k] for c in C) + Tij[(i, j)] + (1 - x_ijk[k][(i, j)]) * M,
                                name=f"TimeUB_{i}_{j}_{k}"
                            )
    #(12)
    for c in C:
        for j in N:
            if nodes[j].type == 'terminal':  # Exclude depot
                for k in KB:
                    model.addConstr(
                        t_jk[j, k] >= Oc[c] * Zcj[c, j] - (1 - f_ck[c, k]) * M,
                        name=f"ReleaseTime_{c}_{j}_{k}"
                    )
    #(13)
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
                t_jk[destination, k] >= t_jk[origin, k]  - (1 - f_ck[c, k]) * M,
                name=f"Sequence_origin_before_destination_indirect_{k}_{c}"
            )

    #=========================================================================================================================
    #  Optimize the Model
    #=========================================================================================================================

    # Update the model with all variables and constraints
    model.update()

    # Set Gurobi parameters
    model.setParam('OutputFlag', True)    # Enable solver output
    model.setParam('TimeLimit', 30)      # Set a time limit of 5 minutes (300 seconds)

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
    visualize_routes(nodes, barges, variables, containers, node_coords,"small.png")

    # Visualize the schedule in gantt chart format of container movements
    visualize_schedule(nodes, barges, variables, containers)


#=============================================================================================================================
#  Run the Program
#=============================================================================================================================

if __name__ == '__main__':
    nodes, arcs, containers, barges, truck, HT, node_coords,depot_to_dummy = construct_network()
    barge_scheduling_problem(nodes, arcs, containers, barges, truck, HT, node_coords,depot_to_dummy)

