import numpy as np
from gurobipy import *
import folium
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
from Visualization import visualize_routes, visualize_schedule_random, visualize_routes_static, visualize_routes_terminals
import random
import logging
from typing import Dict, List, Tuple, Set
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from geopy.distance import geodesic
import folium
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast

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

@dataclass
class AllocationResult:
    barge_id: int
    containers: List[int] = field(default_factory=list)
    route: List[int] = field(default_factory=list)
    departure_time: float = 0.0  # in minutes


def construct_master_routes(
        barges: Dict[int, 'Barge'],
        master_route_sequence: List[int],
        depot_to_dummy: Dict[int, int],
        node_coords: Dict[int, Tuple[float, float]]
) -> Dict[int, List[int]]:
    """
    Constructs the master route for each barge, starting from its origin depot,
    visiting all sea terminals sorted from closest to furthest, and returning to depot_arr.

    Args:
        barges (dict): Dictionary of Barge objects keyed by barge ID.
        master_route_sequence (list): List of sea terminal node IDs.
        depot_to_dummy (dict): Mapping from depot IDs to dummy arrival node IDs.
        node_coords (dict): Coordinates of nodes keyed by node ID, as (latitude, longitude).

    Returns:
        master_routes (dict): Mapping of barge IDs to their sorted master route sequences.
    """
    master_routes = {}
    for barge_id, barge in barges.items():
        origin_depot = barge.origin
        depot_arr = depot_to_dummy[origin_depot]

        # Retrieve coordinates of the origin depot
        origin_coords = node_coords.get(origin_depot)
        if origin_coords is None:
            raise ValueError(f"Coordinates for origin depot {origin_depot} not found in node_coords.")

        # List to store tuples of (terminal_id, distance_from_origin)
        terminals_with_distance = []

        for terminal_id in master_route_sequence:
            terminal_coords = node_coords.get(terminal_id)
            if terminal_coords is None:
                raise ValueError(f"Coordinates for terminal {terminal_id} not found in node_coords.")

            # Calculate the geodesic distance in kilometers
            distance_km = geodesic(origin_coords, terminal_coords).kilometers
            terminals_with_distance.append((terminal_id, distance_km))

        # Sort the terminals based on distance from the origin depot (ascending order)
        sorted_terminals = sorted(terminals_with_distance, key=lambda x: x[1])

        # Extract the sorted terminal IDs
        sorted_terminal_ids = [terminal_id for terminal_id, _ in sorted_terminals]

        # Construct the master route: origin_depot -> sorted_terminals -> depot_arr
        master_route = [origin_depot] + sorted_terminal_ids + [depot_arr]
        master_routes[barge_id] = master_route

    return master_routes

def master_route_index(
    destination: int,
    master_routes: Dict[int, List[int]],
    barges: Dict[int, 'Barge']
) -> int:
    """
    Retrieves the index of the destination in the master route of the first barge.
    Assumes all master routes have the same terminal sequence.

    Args:
        destination (int): Destination node ID.
        master_routes (dict): Mapping of barge IDs to their master route sequences.
        barges (dict): Dictionary of Barge objects.

    Returns:
        index (int): Index of the destination in the master route.
    """
    if not master_routes:
        return 0
    first_barge_id = next(iter(master_routes))
    try:
        return master_routes[first_barge_id].index(destination)
    except ValueError:
        return len(master_routes[first_barge_id])  # If destination not found, place at end

def check_feasibility(
    allocation: 'AllocationResult',
    container: 'Container',
    tentative_route: List[int],
    containers: Dict[int, 'Container'],
    master_routes: Dict[int, List[int]],
    compute_travel_time
) -> Tuple[bool, float]:
    """
    Checks if inserting a container into a barge's route is feasible.

    Args:
        allocation (AllocationResult): Current allocation of the barge.
        container (Container): Container to be inserted.
        tentative_route (list): Tentative route after insertion.
        containers (dict): Dictionary of all containers.
        master_routes (dict): Mapping of barge IDs to their master route sequences.
        compute_travel_time (function): Function to compute travel time between two nodes.

    Returns:
        feasible (bool): True if feasible, False otherwise.
        updated_departure_time (float): Updated departure time if feasible.
    """
    # Initialize departure time
    allocated_containers = allocation.containers + [container.id]
    if any(containers[c].type == 'E' for c in allocated_containers):
        latest_export_release = max(
            containers[c].release_date for c in allocated_containers if containers[c].type == 'E'
        )
        departure_time = latest_export_release
    else:
        departure_time = 0.0

    # Compute arrival times at each node
    arrival_time = departure_time
    current_node = tentative_route[0]

    for next_node in tentative_route[1:]:
        travel_time = compute_travel_time(current_node, next_node)
        arrival_time += travel_time

        # Check time windows for containers destined to next_node
        relevant_containers = [
            c for c in allocated_containers if containers[c].destination == next_node
        ]
        for c_id in relevant_containers:
            container_obj = containers[c_id]
            # If arrival is before opening_date, wait until opening
            if arrival_time < container_obj.opening_date:
                arrival_time = container_obj.opening_date
            # If arrival is after closing_date, infeasible
            if arrival_time > container_obj.closing_date:
                return False, departure_time

        current_node = next_node

    # Update departure time if necessary
    return True, departure_time

def greedy_assign_containers_to_barges(
    containers: Dict[int, Container],
    barges: Dict[int, Barge],
    nodes: Dict[int, Node],
    depot_to_dummy: Dict[int, int],
    master_routes: Dict[int, List[int]],
    node_coords: Dict[int, Tuple[float, float]],
    max_departure_time: float = 196 * 60  # 196 hours in minutes
) -> Tuple[Dict[int, AllocationResult], List[int]]:
    """
    Greedy algorithm to assign containers to barges based on master routes and barge capacities.

    The algorithm ensures that:
    1. All containers picked up from their origin depots are dropped off at their respective destination terminals.
    2. Containers destined for depot_arr are picked up from terminals.
    3. Capacity constraints are respected, considering current load changes due to pickups and drop-offs.

    Args:
        containers (dict): Dictionary of Container objects keyed by container ID.
        barges (dict): Dictionary of Barge objects keyed by barge ID.
        nodes (dict): Dictionary of Node objects keyed by node ID.
        depot_to_dummy (dict): Mapping from depot IDs to dummy arrival node IDs.
        master_routes (dict): Mapping of barge IDs to their master route sequences.
        node_coords (dict): Dictionary mapping node IDs to their (latitude, longitude) coordinates.
        max_departure_time (float): Maximum allowed departure time in minutes.

    Returns:
        allocation (dict): Mapping of barge IDs to AllocationResult objects.
        unassigned_containers (list): List of container IDs not assigned to any barge.
    """
    # Initialize allocation list
    allocation: Dict[int, AllocationResult] = {
        barge.id: AllocationResult(barge.id, containers=[], route=master_routes[barge.id], departure_time=0)
        for barge in barges.values()
    }

    # Initialize onboard containers per barge
    barge_onboard: Dict[int, Set[int]] = {barge.id: set() for barge in barges.values()}

    # Separate containers into origin depot containers and terminal containers
    origin_depot_containers = [c for c in containers.values() if c.origin in depot_to_dummy]
    terminal_containers = [c for c in containers.values() if c.origin not in depot_to_dummy]

    # Sort barges by capacity in descending order
    sorted_barges = sorted(barges.values(), key=lambda x: x.capacity, reverse=True)

    # Helper function to compute travel time
    def compute_travel_time(i: int, j: int) -> float:
        distance = geodesic(node_coords[i], node_coords[j]).kilometers
        speed = 20  # km/h
        return (distance / speed) * 60  # Convert hours to minutes

    # Step 1: Sort origin depot containers by release_date (handling None)
    sorted_origin_depot_containers = sorted(
        origin_depot_containers,
        key=lambda x: (master_route_index(x.destination, master_routes, barges),
                       x.release_date if x.release_date is not None else -float('inf'))
    )

    # Step 2: Sort terminal containers by release_date (handling None)
    sorted_terminal_containers = sorted(
        terminal_containers,
        key=lambda x: (master_route_index(x.destination, master_routes, barges),
                       x.release_date if x.release_date is not None else -float('inf'))
    )

    # Step 3: Assign origin depot containers to barges
    for container in sorted_origin_depot_containers:
        assigned = False
        for barge in sorted_barges:
            master_route = allocation[barge.id].route
            origin = container.origin
            destination = container.destination
            if origin not in master_route or destination not in master_route:
                logging.debug(f"Container {container.id}: Origin or destination not in Barge {barge.id}'s route.")
                continue  # Cannot assign if origin or destination not in route

            origin_idx = master_route.index(origin)
            destination_idx = master_route.index(destination)
            if origin_idx >= destination_idx:
                logging.debug(f"Container {container.id}: Origin occurs after destination in Barge {barge.id}'s route.")
                continue  # Destination should be after origin

            # Check capacity from origin to destination using a temporary simulation
            feasible = True
            temp_load = sum(containers[c_id].size for c_id in barge_onboard[barge.id])
            temp_onboard = set(barge_onboard[barge.id])  # Create a temporary copy

            # Simulate load from origin to destination
            for idx in range(origin_idx, destination_idx + 1):
                node = master_route[idx]

                # Drop off containers at current node
                drop_offs = [c_id for c_id in temp_onboard if containers[c_id].destination == node]
                for c_id in drop_offs:
                    temp_load -= containers[c_id].size
                    temp_onboard.remove(c_id)

                # Pick up containers at current node
                if node == origin:
                    if temp_load + container.size > barge.capacity:
                        feasible = False
                        logging.debug(f"Container {container.id}: Assigning to Barge {barge.id} would exceed capacity at node {node}.")
                        break
                    temp_load += container.size
                    temp_onboard.add(container.id)

                # Check capacity
                if temp_load > barge.capacity:
                    feasible = False
                    logging.debug(f"Barge {barge.id}: Capacity exceeded at node {node} during assignment of Container {container.id}.")
                    break

            if feasible:
                # Assign container to barge
                allocation[barge.id].containers.append(container.id)
                barge_onboard[barge.id].add(container.id)
                assigned = True
                logging.info(f"Assigned Container {container.id} to Barge {barge.id}.")
                break  # Move to next container after successful assignment

        if not assigned:
            logging.warning(f"Could not assign Container {container.id} to any barge from origin depot.")
            logging.warning(f"{container.id} with route Destination: {container.destination}, Origin: {container.origin} is not assigned to any barge.")

    # Step 4: Assign terminal containers to barges
    for container in sorted_terminal_containers:
        assigned = False
        for barge in sorted_barges:
            master_route = allocation[barge.id].route
            origin = container.origin
            destination = container.destination  # Should be depot_arr
            if origin not in master_route or destination not in master_route:
                logging.debug(f"Container {container.id}: Origin or destination not in Barge {barge.id}'s route.")
                continue  # Cannot assign if origin or destination not in route

            origin_idx = master_route.index(origin)
            destination_idx = master_route.index(destination)
            if origin_idx >= destination_idx:
                logging.debug(f"Container {container.id}: Origin occurs after destination in Barge {barge.id}'s route.")
                continue  # Destination should be after origin

            # Check capacity from origin to destination using a temporary simulation
            feasible = True
            temp_load = sum(containers[c_id].size for c_id in barge_onboard[barge.id])
            temp_onboard = set(barge_onboard[barge.id])  # Create a temporary copy

            # Simulate load from origin to destination
            for idx in range(origin_idx, destination_idx + 1):
                node = master_route[idx]

                # Drop off containers at current node
                drop_offs = [c_id for c_id in temp_onboard if containers[c_id].destination == node]
                for c_id in drop_offs:
                    temp_load -= containers[c_id].size
                    temp_onboard.remove(c_id)

                # Pick up containers at current node
                if node == origin:
                    if temp_load + container.size > barge.capacity:
                        feasible = False
                        logging.debug(f"Container {container.id}: Assigning to Barge {barge.id} would exceed capacity at node {node}.")
                        break
                    temp_load += container.size
                    temp_onboard.add(container.id)

                # Check capacity
                if temp_load > barge.capacity:
                    feasible = False
                    logging.debug(f"Barge {barge.id}: Capacity exceeded at node {node} during assignment of Container {container.id}.")
                    break

            if feasible:
                # Assign container to barge
                allocation[barge.id].containers.append(container.id)
                barge_onboard[barge.id].add(container.id)
                assigned = True
                logging.info(f"Assigned Container {container.id} to Barge {barge.id}.")
                break  # Move to next container after successful assignment

        if not assigned:
            logging.warning(f"Could not assign Container {container.id} to any barge from terminal.")
            logging.warning(f"{container.id} with route Destination: {container.destination}, Origin: {container.origin} is not assigned to any barge.")

    # Step 5: Identify unassigned containers
    assigned_containers = set()
    for alloc in allocation.values():
        assigned_containers.update(alloc.containers)
    final_unassigned_containers = [c.id for c in containers.values() if c.id not in assigned_containers]

    if final_unassigned_containers:
        logging.info(f"{len(final_unassigned_containers)} containers unassigned to barges and will be assigned to trucks.")

    return allocation, final_unassigned_containers


def random_color() -> str:
    """
    Generates a random color in HEX format.

    Returns:
        str: HEX color string.
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))



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

def random_generation_containers(nodes, node_coords,container_amount, buffer_time=24*60):
     # Number of containers to generate
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

            origin = 0 # Random depot origin (0-4)
            destination = random.randint(5, 24)  # Random terminal destination (5-24)

        else:  # Import containers
            release_date = None
            # Opening date within first 24 hours
            opening_date = random.randint(0, 24*60)  # 0 to 1440 minutes
            # Closing date is at least buffer_time after opening_date
            max_closing_date = 24*60 + 172*60  # Up to ~3 days
            closing_date = opening_date + buffer_time + random.randint(0, max_closing_date - (opening_date + buffer_time))

            origin = random.randint(5, 24)  # Random terminal origin (5-24)
            destination = 25  # Random depot arrival destination (25-29)

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

def construct_network(container_amount):
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
        # 1: Node(1, 'depot'),  #  (e.g., Tilburg Depot)
        # 2: Node(2, 'depot'),  # (e.g., Eindhoven Depot)
        # 3: Node(3, 'depot'),  #  (e.g., Nijmegen Depot)
        # 4: Node(4, 'depot'),  # (e.g., Utrecht Depot)

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

        25: Node(25, 'depot_arr')  # Dummy arrival for Veghel Depot
        # 26: Node(26, 'depot_arr'),  # Tilburg
        # 27: Node(27, 'depot_arr'),  # Eindhoven
        # 28: Node(28, 'depot_arr'),  # Nijmegen
        # 29: Node(29, 'depot_arr')   # Utrecht
    }

    # Define coordinates for each node (latitude, longitude)
    # Updated depot coordinates to reflect inland locations like Veghel
    node_coords = {
        # Depots (Inland Locations)
        0: (51.5022, 5.6875),  # Veghel Depot
        # 1: (51.5667, 5.0689),  # Tilburg Depot
        # 2: (51.4416, 5.4697),  # Eindhoven Depot
        # 3: (51.8126, 5.8372),  # Nijmegen Depot
        # 4: (52.0907, 5.1214),  # Utrecht Depot

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
        25: (51.5022, 5.6875)  # Veghel Depot_arr
        # 26: (51.5667, 5.0689),  # Tilburg Depot_arr
        # 27: (51.4416, 5.4697),  # Eindhoven Depot_arr
        # 28: (51.8126, 5.8372),  # Nijmegen Depot_arr
        # 29: (52.0907, 5.1214)  # Utrecht Depot_arr
    }

    depot_to_dummy = {
        0: 25  # depot 0 matches with dummy node 25
        # 1: 26,
        # 2: 27,
        # 3: 28,
        # 4: 29
    }

    # Define containers with their attributes
    containers_data = random_generation_containers(nodes, node_coords,container_amount)

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

    # Define trucks with their cost per container
    # HT = {1: 140,
    #       2: 200}  # You can add more truck IDs and their costs if needed




    return nodes, arcs, containers, node_coords, depot_to_dummy



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

def barge_scheduling_problem(
        nodes, arcs, containers, barges, HT, node_coords, depot_to_dummy,
        master_routes, greedy_allocation, unassigned_containers,IS, Count, Analysis, k , l, changed_values
    ):
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
    M2 = 100

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
    Tij = {(arc.origin, arc.destination): (1-k*0.1)*arc.travel_time for arc in arcs}  # Tij: Travel times between nodes

    cheat = np.ones(20)
    cheat[-1] = 0

    L = 15     # Handling time per container in minutes (e.g., loading/unloading time)
    gamma = 100 + l * 5 + 5 - 5 * cheat[l] # Penalty cost for visiting sea terminals
    changed_values.update({"gamma": gamma})
    changed_values.update({"Tij": Tij})
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
            p_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{j}_{k}")
            d_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_{j}_{k}")

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
        quicksum(x_ijk[k][(i, j)] * HBk[k] for k in KB for (i, j) in x_ijk[k] if
                 nodes[i].type == "depot" and nodes[j].type == "terminal") +
        quicksum(Tij[(i, j)] * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k]) +
        quicksum(gamma * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k] if nodes[i].type == "terminal"),
        GRB.MINIMIZE
    )

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
                t_jk[destination, k] >= t_jk[origin, k] - (1 - f_ck[c, k]) * M2,
                name=f"Sequence_origin_before_destination_indirect_{k}_{c}"
            )

    if IS == True:
        # Iterate through the greedy allocation and set variable starts
        for k, allocation_result in greedy_allocation.items():
            # Set x_ijk variables based on the route
            route = allocation_result.route
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                if (from_node, to_node) in x_ijk[k]:
                    x_ijk[k][(from_node, to_node)].Start = 1
                else:
                    # If the arc is not part of the route, set to 0
                    x_ijk[k][(from_node, to_node)].Start = 0
                    pass  # Gurobi initializes binary variables to 0 by default

            # Assign containers to barges
            for c in allocation_result.containers:
                f_ck[c, k].Start = 1
                # Ensure container is not assigned to truck
                f_ck[c, 'T'].Start = 0

        # Assign unassigned containers to trucks
        for c in unassigned_containers:
            f_ck[c, 'T'].Start= 1
            # Ensure container is not assigned to any barge
            for k in KB:
                f_ck[c, k].Start = 0


    #=========================================================================================================================
    #  Optimize the Model
    #=========================================================================================================================

    # Update the model with all variables and constraints
    model.update()
    #print models status
    print(model.Status)

    # Set Gurobi parameters
    model.setParam('OutputFlag', True)
    model.setParam('StartNodeLimit',2000)
    # Enable solver output
    model.setParam("MIPFocus", 3)  # Emphasize feasibility")
    model.setParam("presolve", 2)
    model.setParam("heuristics", 0.7)
    model.setParam("Cuts", 3)
    model.setParam("MIRCuts", 2)
    model.setParam('TimeLimit', 20000)      # Set a time limit of 5 minutes (300 seconds)
    model.setParam('MIPGap', 0.05)
    # Start the optimization process
    model.optimize()

    #solution = {}
    #for var in model.getVars():
        #solution[var.VarName] = var.X
        # Convert solution to DataFrame


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

    string = Analysis.replace(" ", "_")
    output_base = f"Sensitivity_analysis_" + string + "_" + str(Count) + ".png"
    # Print the optimization results: objective value, container allocations, and barge routes
    print_model_result(model, variables, barges, containers)

    # Visualize the barge and truck routes on a map
    # visualize_routes_static(nodes, barges, variables, containers, node_coords,output_filename_full=output_base+"route_including_depot.png")
    # visualize_routes(nodes, barges, variables, containers, node_coords,file_name=output_base+"interactive_route.html")
    #
    # # Visualize the schedule in gantt chart format of container movements
    visualize_schedule_random(nodes, barges, variables, containers, Count, Analysis, output_file=output_base)
    # visualize_routes_terminals(nodes, barges, variables, containers, node_coords, output_file=output_base+"route_terminals.png")

    if model.status == GRB.OPTIMAL:
        objective_value = model.objVal
        Bound = model.ObjBound
        Gap = model.MIPGap
        OBG = [objective_value, Bound, Gap]
    else:
        objective_value = model.objVal
        Bound = model.ObjBound
        Gap = model.MIPGap
        OBG = [objective_value, Bound, Gap]

    return OBG, variables, changed_values


def write_results_to_csv(file_name, Analysis, count, objective_value, variables, Changed_value):
    # Prepare results as a dictionary
    results = {
        "Count": [count],
        "Analysis": [Analysis],
        "Objective Value": [objective_value],
        "Variables": [variables],
        "Changed value": [Changed_value]
    }

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Append to the file or create it if it doesn't exist
    if not os.path.isfile(file_name):
        # If file doesn't exist, write with header
        results_df.to_csv(file_name, mode="w", index=False)
    else:
        # If file exists, append without header
        results_df.to_csv(file_name, mode="a", header=False, index=False)

    print(f"Results written to {file_name} (Iteration {count}).")



def execute_gurobi_optimization(nr_c, IS, i, j, k, l, m, Analysis, file_name="results.csv"):
    """
    Executes the greedy algorithm, optimizes with Gurobi, and maintains consistent output and visualization.
    """
    # Step 1: Execute Greedy Assignment
    container_amount = nr_c
    nodes, arcs, containers, node_coords, depot_to_dummy = construct_network(container_amount=container_amount)
    changed_values = {}
    cheat = np.ones(40)
    cheat[-1] = 0

    HT = {1: 175,
          2: 195}

    barges_data = [
        (1, 15+m*1-cheat[m]*6+1, 250 + 150*j+150, 0),  # Barge 1: Capacity=104, Fixed Cost=3600,
        (2, 11+m*1-cheat[m]*6+1, 230 + 120*j+120, 0),
        (3, 7+m*1-cheat[m]*6+1, 238 + 72*j+72, 0)
    ]
    barges = {barge_id: Barge(barge_id, capacity, fixed_cost, origin)
              for barge_id, capacity, fixed_cost, origin in barges_data}
    changed_values.update({"barges": barges_data})
    # truck = Truck(cost_per_container=HT)

    #CHANGE PARAMETERS
    Count = max(i, j, k, l, m)


    # Increase the truck costs in HT
    for truck_id, truck_cost in HT.items():
        HT[truck_id] = HT[truck_id] + 15 * i - cheat[i] * 65 + 15
    changed_values.update({"Trucks": HT})
    # Update the truck object with the new HT values

    # Check the updated truck costs
    print(f"Updated truck costs: {HT}", f"Updated barge costs: {barges[1].fixed_cost, barges[2].fixed_cost, barges[3].fixed_cost}", f"Count = {Count}")


    master_route_sequence = [
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24  # Sea terminals
    ]

    master_routes = construct_master_routes(barges, master_route_sequence, depot_to_dummy, node_coords)

    greedy_allocation, unassigned_containers = greedy_assign_containers_to_barges(
        containers, barges, nodes, depot_to_dummy, master_routes, node_coords
    )

    # Step 2: Optimize with Gurobi using Greedy Assignment as MIP Start
    objective_value, variables, changed_values = barge_scheduling_problem(
        nodes, arcs, containers, barges, HT, node_coords, depot_to_dummy,
        master_routes, greedy_allocation, unassigned_containers, IS, Count, Analysis, k, l, changed_values
    )

    #Step 3: Output Results
    if IS == True:
        print("\nGreedy Allocation Results:")
        for barge_id, result in greedy_allocation.items():
            print(f"Barge {barge_id} has {len(result.containers)} containers:")
            print(f"Containers: {result.containers}")
            print(f"Route: {result.route}")
            print(f"Departure Time: {result.departure_time} minutes\n")

        if unassigned_containers:
            print(f"Containers assigned to trucks: {len(unassigned_containers)}")
            print(f"Container IDs: {unassigned_containers}")


    # Use `write_results_to_csv`
    write_results_to_csv(file_name, Analysis, Count, objective_value, variables, changed_values)


if __name__ == "__main__":
    number_containers = 50
    IS = False
    file_name = "results_pc.csv"

    #execute_gurobi_optimization(number_containers, IS, 19, -1, 0, -1, -1, "Truck costs MEGA Tronic", file_name)

    #execute_gurobi_optimization(number_containers, IS, 0, 0, 0, 0, 0, "Base Case", file_name)
    # for i in range(9,20):
    #     execute_gurobi_optimization(number_containers, IS, i, -1, 0, -1, -1, "Truck costs", file_name)
    # for i in range(20):
    #     execute_gurobi_optimization(number_containers, IS, -1, i, 0, -1, -1, "Barge costs", file_name)
    # for i in range(21,30):
    #     execute_gurobi_optimization(number_containers, IS, -1, -1, 0, -1, i, "Barge capacity", file_name)
    # for i in range(20):
    #     execute_gurobi_optimization(number_containers, IS, -1, -1, i, -1, -1, "Travel time", file_name)
    # for i in range(16, 20):
    #     execute_gurobi_optimization(number_containers, IS, -1, -1, 0, i, -1, "Penalty cost", file_name)

import pandas as pd
import ast
import matplotlib.pyplot as plt

state = 1  # Set to 1 to run this section

if state:
    file_name = "results_pc.csv"

    # Read the CSV file (adjust header and skiprows as needed)
    df = pd.read_csv(file_name, header=None, skiprows=1)

    # Manually set column names (adjust these as needed)
    df.columns = ['Count', 'Analysis', 'Objective Value', 'Variables', 'Changed value']
    print(df.head())

    # Define safe_eval to safely convert string representations of lists
    def safe_eval(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) else val
        except (ValueError, SyntaxError):
            return None

    # Convert the 'Objective Value' column from string to an actual list
    df['Objective Value'] = df['Objective Value'].apply(safe_eval)

    # Extract the first element of the 'Objective Value' list as the objective value
    df['ObjValue_First'] = df['Objective Value'].apply(
        lambda arr: arr[0] if isinstance(arr, list) and len(arr) > 0 else None
    )

    # Extract the gap (assumed to be the third element in the list) and convert to percent
    df['Gap Percent'] = df['Objective Value'].apply(
        lambda arr: 100 * arr[2] if isinstance(arr, list) and len(arr) > 2 and arr[2] is not None else None
    )

    # Debug: check the converted values
    print(df[['Objective Value', 'ObjValue_First', 'Gap Percent']].head(10))

    # *** Define two row ranges based on indices ***
    # Adjust these indices to select the rows corresponding to your desired ranges.
    # For example, if rows 205:221 correspond to the range for 0-15 and rows 221:227 correspond to the range 16-19:
    range1 = df.iloc[144:164]  # first range (e.g., 0–15)
    range2 = df.iloc[226:235]  # second range (e.g., 16–19)

    # Filter out rows where 'Analysis' contains "Barge capacity"
    #range1 = range1[~range1['Analysis'].str.contains('Barge capacity', na=False)]
    #range2 = range2[~range2['Analysis'].str.contains('Barge capacity', na=False)]

    # Combine the two ranges into one DataFrame for plotting
    plot_df = pd.concat([range1, range2])

    # Extract values for plotting
    count = plot_df['Count']
    obj_value = plot_df['ObjValue_First']
    gap_percent = plot_df['Gap Percent']

    # Create a larger plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Objective Value on the left y-axis
    color_obj = 'red'
    ln1 = ax1.plot(count, obj_value, marker='s', linestyle='-', color=color_obj, label='Objective Value')
    ax1.set_xlabel('Count', color='black')
    ax1.set_ylabel('Objective Value', color='black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax1.grid(True)
    #ax1.set_ylim(2800, 6000)

    # Create a second y-axis for Gap Percent
    ax2 = ax1.twinx()
    color_gap = 'blue'
    ln2 = ax2.plot(count, gap_percent, marker='o', linestyle='-', color=color_gap, label='Gap Percent')
    ax2.set_ylabel('Gap in Percent', color='black')
    ax2.tick_params(axis='y', colors='black')
    #ax2.set_ylim(32, 46)

    # Combine legends from both axes
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='best')

    plt.title('Objective Value and Gap Percent vs. Count', color='black')
    plt.tight_layout()
    plt.show()










