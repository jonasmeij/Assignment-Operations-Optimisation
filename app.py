# interactive_network_creator_with_optimizer_pixel.py

import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import json
import os
from math import sqrt
from gurobipy import *
import webbrowser

# Define data classes
class Node:
    def __init__(self, node_id, x, y, node_type='terminal'):
        self.id = node_id
        self.x = x  # Pixel x-coordinate on the map
        self.y = y  # Pixel y-coordinate on the map
        self.type = node_type  # 'depot' or 'terminal'

class Container:
    def __init__(self, container_id, size, release_date, opening_date, closing_date, origin, destination, container_type):
        self.id = container_id
        self.size = size
        self.release_date = release_date
        self.opening_date = opening_date
        self.closing_date = closing_date
        self.origin = origin
        self.destination = destination
        self.type = container_type  # 'I' or 'E'

class Barge:
    def __init__(self, barge_id, capacity, fixed_cost):
        self.id = barge_id
        self.capacity = capacity
        self.fixed_cost = fixed_cost

class Truck:
    def __init__(self, truck_id, cost_per_container):
        self.id = truck_id
        self.cost_per_container = cost_per_container

# Main Application Class
class NetworkCreatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive Network Creator with Optimizer (Pixel Coordinates)")

        # Initialize data storage
        self.nodes = {}
        self.containers = {}
        self.barges = {}
        self.trucks = {}
        self.next_node_id = 0
        self.next_container_id = 1
        self.next_barge_id = 1
        self.next_truck_id = 1

        # Load map
        self.load_map()

        # Create menus
        self.create_menus()

        # Create buttons for data input and optimization
        self.create_buttons()

        # Bind click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def load_map(self):
        # Prompt user to select a PNG map
        map_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG Files", "*.png")])
        if not map_path:
            messagebox.showerror("Error", "No map selected. Exiting.")
            self.master.destroy()
            return

        # Load the image
        self.map_image = Image.open(map_path)
        self.map_photo = ImageTk.PhotoImage(self.map_image)

        # Create Canvas
        self.canvas = tk.Canvas(self.master, width=self.map_photo.width(), height=self.map_photo.height(), bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)

    def create_menus(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Network", command=self.save_network)
        file_menu.add_command(label="Load Network", command=self.load_network)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def create_buttons(self):
        # Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Button to add containers
        add_container_btn = tk.Button(button_frame, text="Add Containers", command=self.add_container)
        add_container_btn.pack(pady=5, fill=tk.X)

        # Button to add barge
        add_barge_btn = tk.Button(button_frame, text="Add Barge", command=self.add_barge)
        add_barge_btn.pack(pady=5, fill=tk.X)

        # Button to add truck
        add_truck_btn = tk.Button(button_frame, text="Add Truck", command=self.add_truck)
        add_truck_btn.pack(pady=5, fill=tk.X)

        # Button to view data
        view_data_btn = tk.Button(button_frame, text="View Data", command=self.view_data)
        view_data_btn.pack(pady=5, fill=tk.X)

        # Button to solve the optimization model
        solve_model_btn = tk.Button(button_frame, text="Solve Model", command=self.solve_model)
        solve_model_btn.pack(pady=20, fill=tk.X)

    def on_canvas_click(self, event):
        x, y = event.x, event.y

        # Prompt user for node type
        node_type = simpledialog.askstring("Node Type", "Enter node type ('depot' or 'terminal'):")
        if node_type not in ['depot', 'terminal']:
            messagebox.showerror("Invalid Type", "Node type must be 'depot' or 'terminal'.")
            return

        node_id = self.next_node_id
        self.next_node_id += 1
        node = Node(node_id, x, y, node_type)
        self.nodes[node_id] = node
        self.draw_node(node)
        messagebox.showinfo("Node Added", f"Node {node_id} added as {node_type}.")

    def draw_node(self, node):
        # Define colors based on node type
        color = 'blue' if node.type == 'depot' else 'green'
        radius = 5
        self.canvas.create_oval(node.x - radius, node.y - radius, node.x + radius, node.y + radius, fill=color, outline='black')
        self.canvas.create_text(node.x, node.y - 10, text=str(node.id), fill='black', font=('Arial', 10, 'bold'))

    def get_node_ids_by_type(self, node_type):
        """
        Returns a list of node IDs filtered by the specified node type.
        Args:
            node_type (str): 'depot' or 'terminal'
        Returns:
            List of node IDs matching the type.
        """
        return [str(node.id) for node in self.nodes.values() if node.type == node_type]

    def add_container(self):
        if not self.nodes:
            messagebox.showerror("No Nodes", "Please add nodes before adding containers.")
            return

        # Create a new window for container input
        container_window = tk.Toplevel(self.master)
        container_window.title("Add Containers")
        container_window.geometry("800x400")  # Adjust size as needed

        # Create a canvas and a vertical scrollbar for the container inputs
        canvas = tk.Canvas(container_window, borderwidth=0, background="#f0f0f0")
        scrollbar = tk.Scrollbar(container_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, background="#f0f0f0")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Initialize a list to keep track of container entries
        container_entries = []

        # Define table headers
        headers = ["Size", "Release Date", "Opening Date", "Closing Date",
                   "Origin Node ID", "Destination Node ID", "Type", "Action"]
        for col, header in enumerate(headers):
            tk.Label(scrollable_frame, text=header, borderwidth=1, relief="solid", bg="#d3d3d3").grid(row=0, column=col, padx=1, pady=1, sticky='nsew')

        # Function to add a new container input row
        def add_container_row():
            row = len(container_entries) + 1  # Start from row 1 as row 0 has headers

            # Size
            size_entry = tk.Entry(scrollable_frame)
            size_entry.grid(row=row, column=0, padx=1, pady=1, sticky='nsew')

            # Release Date
            release_entry = tk.Entry(scrollable_frame)
            release_entry.grid(row=row, column=1, padx=1, pady=1, sticky='nsew')

            # Opening Date
            opening_entry = tk.Entry(scrollable_frame)
            opening_entry.grid(row=row, column=2, padx=1, pady=1, sticky='nsew')

            # Closing Date
            closing_entry = tk.Entry(scrollable_frame)
            closing_entry.grid(row=row, column=3, padx=1, pady=1, sticky='nsew')

            # Origin node
            origin_var = tk.StringVar(scrollable_frame)
            origin_menu = tk.OptionMenu(scrollable_frame, origin_var, *self.get_node_ids_by_type('depot'))
            origin_menu.grid(row=row, column=4, padx=1, pady=1, sticky='nsew')

            # Destination node
            destination_var = tk.StringVar(scrollable_frame)
            destination_menu = tk.OptionMenu(scrollable_frame, destination_var, *self.get_node_ids_by_type('terminal'))
            destination_menu.grid(row=row, column=5, padx=1, pady=1, sticky='nsew')

            # Container type
            type_var = tk.StringVar(scrollable_frame)
            type_menu = tk.OptionMenu(scrollable_frame, type_var, 'E', 'I')
            type_menu.grid(row=row, column=6, padx=1, pady=1, sticky='nsew')
            type_var.set('E')  # Set default type

            # Action (Delete button)
            delete_btn = tk.Button(scrollable_frame, text="Delete", command=lambda: delete_container_row(row))
            delete_btn.grid(row=row, column=7, padx=1, pady=1, sticky='nsew')

            # Define the on_type_change function
            def on_type_change(*args):
                container_type = type_var.get()
                # Update origin and destination menus based on container type
                if container_type == 'E':
                    # Export: Origin = depot, Destination = terminal
                    origin_menu['menu'].delete(0, 'end')
                    for depot_id in self.get_node_ids_by_type('depot'):
                        origin_menu['menu'].add_command(label=depot_id, command=lambda value=depot_id: origin_var.set(value))
                    destination_menu['menu'].delete(0, 'end')
                    for terminal_id in self.get_node_ids_by_type('terminal'):
                        destination_menu['menu'].add_command(label=terminal_id, command=lambda value=terminal_id: destination_var.set(value))
                elif container_type == 'I':
                    # Import: Origin = terminal, Destination = depot
                    origin_menu['menu'].delete(0, 'end')
                    for terminal_id in self.get_node_ids_by_type('terminal'):
                        origin_menu['menu'].add_command(label=terminal_id, command=lambda value=terminal_id: origin_var.set(value))
                    destination_menu['menu'].delete(0, 'end')
                    for depot_id in self.get_node_ids_by_type('depot'):
                        destination_menu['menu'].add_command(label=depot_id, command=lambda value=depot_id: destination_var.set(value))

            # Bind the type_var to the on_type_change function
            type_var.trace('w', on_type_change)

            # Append the entries to the list
            container_entries.append({
                'size': size_entry,
                'release': release_entry,
                'opening': opening_entry,
                'closing': closing_entry,
                'origin': origin_var,
                'destination': destination_var,
                'type': type_var
            })

        # Function to delete a container row
        def delete_container_row(row):
            # Disable editing for simplicity (optional: implement actual row deletion)
            for widget in scrollable_frame.grid_slaves(row=row):
                widget.grid_forget()

        # Button to add more container rows
        add_another_btn = tk.Button(scrollable_frame, text="Add Another Container", command=add_container_row)
        add_another_btn.grid(row=1000, column=0, padx=5, pady=10, sticky='w')  # Use a high row number to place it at the bottom

        # Submit button
        submit_btn = tk.Button(scrollable_frame, text="Submit All Containers", command=lambda: self.submit_containers(container_window, container_entries))
        submit_btn.grid(row=1001, column=0, columnspan=8, pady=20)  # Adjust columnspan as per headers

        # Add the first container row
        add_container_row()

    def submit_containers(self, window, container_entries):
        try:
            for idx, entry in enumerate(container_entries, start=1):
                size = entry['size'].get().strip()
                release = entry['release'].get().strip()
                opening = entry['opening'].get().strip()
                closing = entry['closing'].get().strip()
                origin = entry['origin'].get().strip()
                destination = entry['destination'].get().strip()
                container_type = entry['type'].get().strip()

                # Validate size
                if not size:
                    raise ValueError(f"Container {idx}: Size is required.")
                size = float(size)

                # Validate container type
                if container_type not in ['E', 'I']:
                    raise ValueError(f"Container {idx}: Type must be 'E' or 'I'.")

                # Validate origin and destination
                origin = int(origin)
                destination = int(destination)
                if container_type == 'E':
                    if self.nodes[origin].type != 'depot' or self.nodes[destination].type != 'terminal':
                        raise ValueError(f"Container {idx}: Export containers must originate from a depot and be destined to a terminal.")
                elif container_type == 'I':
                    if self.nodes[origin].type != 'terminal' or self.nodes[destination].type != 'depot':
                        raise ValueError(f"Container {idx}: Import containers must originate from a terminal and be destined to a depot.")
                    release = None  # For import containers, release date is None

                # Validate release, opening, and closing dates
                if container_type == 'E':
                    if not release:
                        raise ValueError(f"Container {idx}: Release date is required for export containers.")
                    release = int(release)
                else:
                    release = None  # Ensure release is None for import containers

                if not opening:
                    raise ValueError(f"Container {idx}: Opening date is required.")
                opening = int(opening)

                if not closing:
                    raise ValueError(f"Container {idx}: Closing date is required.")
                closing = int(closing)

                # Create and add the container
                container_id = self.next_container_id
                self.next_container_id += 1
                container = Container(container_id, size, release, opening, closing, origin, destination, container_type)
                self.containers[container_id] = container

        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
            return

        messagebox.showinfo("Containers Added", f"{len(container_entries)} container(s) added successfully.")
        window.destroy()

    def add_barge(self):
        # Create a new window for barge input
        barge_window = tk.Toplevel(self.master)
        barge_window.title("Add Barge")

        # Barge attributes
        tk.Label(barge_window, text="Capacity:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        capacity_entry = tk.Entry(barge_window)
        capacity_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        tk.Label(barge_window, text="Fixed Cost:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        fixed_cost_entry = tk.Entry(barge_window)
        fixed_cost_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Submit button
        submit_btn = tk.Button(barge_window, text="Add", command=lambda: self.submit_barge(barge_window, capacity_entry.get(), fixed_cost_entry.get()))
        submit_btn.grid(row=2, column=0, columnspan=2, pady=10)

    def submit_barge(self, window, capacity, fixed_cost):
        try:
            capacity = float(capacity)
            fixed_cost = float(fixed_cost)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid barge data.")
            return

        barge_id = self.next_barge_id
        self.next_barge_id += 1
        barge = Barge(barge_id, capacity, fixed_cost)
        self.barges[barge_id] = barge
        messagebox.showinfo("Barge Added", f"Barge {barge_id} added.")
        window.destroy()

    def add_truck(self):
        # Create a new window for truck input
        truck_window = tk.Toplevel(self.master)
        truck_window.title("Add Truck")

        # Truck attributes
        tk.Label(truck_window, text="Cost per Container:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        cost_entry = tk.Entry(truck_window)
        cost_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Submit button
        submit_btn = tk.Button(truck_window, text="Add", command=lambda: self.submit_truck(truck_window, cost_entry.get()))
        submit_btn.grid(row=1, column=0, columnspan=2, pady=10)

    def submit_truck(self, window, cost):
        try:
            cost = float(cost)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid cost.")
            return

        truck_id = self.next_truck_id
        self.next_truck_id += 1
        truck = Truck(truck_id, cost)
        self.trucks[truck_id] = truck
        messagebox.showinfo("Truck Added", f"Truck {truck_id} added.")
        window.destroy()

    def view_data(self):
        # Create a new window to display all data
        data_window = tk.Toplevel(self.master)
        data_window.title("Network Data")

        # Create a notebook with tabs
        notebook = ttk.Notebook(data_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Nodes Tab
        nodes_frame = ttk.Frame(notebook)
        notebook.add(nodes_frame, text="Nodes")
        nodes_text = tk.Text(nodes_frame, width=80, height=20)
        nodes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nodes_scroll = tk.Scrollbar(nodes_frame, command=nodes_text.yview)
        nodes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        nodes_text.config(yscrollcommand=nodes_scroll.set)
        for node in self.nodes.values():
            nodes_text.insert(tk.END, f"ID: {node.id}, Type: {node.type}, Pixel Coords: ({node.x}, {node.y})\n")

        # Containers Tab
        containers_frame = ttk.Frame(notebook)
        notebook.add(containers_frame, text="Containers")
        containers_text = tk.Text(containers_frame, width=80, height=20)
        containers_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        containers_scroll = tk.Scrollbar(containers_frame, command=containers_text.yview)
        containers_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        containers_text.config(yscrollcommand=containers_scroll.set)
        for container in self.containers.values():
            containers_text.insert(tk.END, f"ID: {container.id}, Size: {container.size}, Type: {container.type}, "
                                           f"Origin: {container.origin}, Destination: {container.destination}, "
                                           f"Release: {container.release_date}, Opening: {container.opening_date}, "
                                           f"Closing: {container.closing_date}\n")

        # Barges Tab
        barges_frame = ttk.Frame(notebook)
        notebook.add(barges_frame, text="Barges")
        barges_text = tk.Text(barges_frame, width=80, height=10)
        barges_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barges_scroll = tk.Scrollbar(barges_frame, command=barges_text.yview)
        barges_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        barges_text.config(yscrollcommand=barges_scroll.set)
        for barge in self.barges.values():
            barges_text.insert(tk.END, f"ID: {barge.id}, Capacity: {barge.capacity}, Fixed Cost: {barge.fixed_cost}\n")

        # Trucks Tab
        trucks_frame = ttk.Frame(notebook)
        notebook.add(trucks_frame, text="Trucks")
        trucks_text = tk.Text(trucks_frame, width=80, height=10)
        trucks_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trucks_scroll = tk.Scrollbar(trucks_frame, command=trucks_text.yview)
        trucks_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        trucks_text.config(yscrollcommand=trucks_scroll.set)
        for truck in self.trucks.values():
            trucks_text.insert(tk.END, f"ID: {truck.id}, Cost per Container: {truck.cost_per_container}\n")

    def save_network(self):
        # Prompt user to select a file to save
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not save_path:
            return

        # Prepare data
        data = {
            "nodes": {node_id: {"x": node.x, "y": node.y, "type": node.type}
                      for node_id, node in self.nodes.items()},
            "containers": {container_id: {
                "size": container.size,
                "release_date": container.release_date,
                "opening_date": container.opening_date,
                "closing_date": container.closing_date,
                "origin": container.origin,
                "destination": container.destination,
                "type": container.type
            } for container_id, container in self.containers.items()},
            "barges": {barge_id: {
                "capacity": barge.capacity,
                "fixed_cost": barge.fixed_cost
            } for barge_id, barge in self.barges.items()},
            "trucks": {truck_id: {
                "cost_per_container": truck.cost_per_container
            } for truck_id, truck in self.trucks.items()},
            "next_ids": {
                "node": self.next_node_id,
                "container": self.next_container_id,
                "barge": self.next_barge_id,
                "truck": self.next_truck_id
            }
        }

        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Saved", f"Network data saved to {save_path}.")

    def load_network(self):
        # Prompt user to select a JSON file
        load_path = filedialog.askopenfilename(title="Load Network Data", filetypes=[("JSON Files", "*.json")])
        if not load_path:
            return

        if not os.path.exists(load_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return

        # Load data from JSON
        with open(load_path, 'r') as f:
            data = json.load(f)

        # Clear existing data
        self.clear_data()

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node_id = int(node_id)
            node = Node(node_id, node_data["x"], node_data["y"], node_data["type"])
            self.nodes[node_id] = node
            self.draw_node(node)

        # Load containers
        for container_id, container_data in data.get("containers", {}).items():
            container_id = int(container_id)
            container = Container(container_id, container_data["size"],
                                  container_data["release_date"],
                                  container_data["opening_date"],
                                  container_data["closing_date"],
                                  container_data["origin"],
                                  container_data["destination"],
                                  container_data["type"])
            self.containers[container_id] = container

        # Load barges
        for barge_id, barge_data in data.get("barges", {}).items():
            barge_id = int(barge_id)
            barge = Barge(barge_id, barge_data["capacity"], barge_data["fixed_cost"])
            self.barges[barge_id] = barge

        # Load trucks
        for truck_id, truck_data in data.get("trucks", {}).items():
            truck_id = int(truck_id)
            truck = Truck(truck_id, truck_data["cost_per_container"])
            self.trucks[truck_id] = truck

        # Load next IDs
        self.next_node_id = data.get("next_ids", {}).get("node", self.next_node_id)
        self.next_container_id = data.get("next_ids", {}).get("container", self.next_container_id)
        self.next_barge_id = data.get("next_ids", {}).get("barge", self.next_barge_id)
        self.next_truck_id = data.get("next_ids", {}).get("truck", self.next_truck_id)

        messagebox.showinfo("Loaded", f"Network data loaded from {load_path}.")

    def clear_data(self):
        # Clear all data
        self.nodes.clear()
        self.containers.clear()
        self.barges.clear()
        self.trucks.clear()
        self.next_node_id = 0
        self.next_container_id = 1
        self.next_barge_id = 1
        self.next_truck_id = 1

        # Clear canvas (remove all except the map image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)

    def show_about(self):
        messagebox.showinfo("About", "Interactive Network Creator with Optimizer\nDeveloped with Tkinter and Gurobi.")

    def solve_model(self):
        if not self.nodes:
            messagebox.showerror("No Nodes", "Please add nodes before solving the model.")
            return
        if not self.containers:
            messagebox.showerror("No Containers", "Please add containers before solving the model.")
            return
        if not self.barges:
            messagebox.showerror("No Barges", "Please add barges before solving the model.")
            return
        if not self.trucks:
            messagebox.showerror("No Trucks", "Please add trucks before solving the model.")
            return

        try:
            # Construct the optimization model
            nodes_dict, arcs_list = self.construct_arcs()
            model = self.build_and_solve_model(nodes_dict, arcs_list)
            if model is None:
                return  # Model was infeasible or an error occurred
            # Extract variables
            variables = self.extract_variables(model, nodes_dict, arcs_list)
            # Display results
            self.display_results(model, variables)
            # Visualize routes
            self.visualize_routes(None, variables)  # No need to save as image
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during optimization:\n{str(e)}")


    def build_and_solve_model(self, nodes, arcs):
        """
        Builds and solves the optimization model using Gurobi.
        Args:
            nodes (dict): Dictionary of Node objects keyed by node ID.
            arcs (list): List of Arc tuples representing possible routes (origin, destination, travel_time).
        Returns:
            model (gurobipy.Model): The optimized Gurobi model, or None if infeasible.
        """
        model = Model("TransportationOptimization")

        # Big M
        M = 10000  # A large constant used in Big M method for conditional constraints

        # Define sets
        N = list(nodes.keys())                         # Set of all node IDs
        C = list(self.containers.keys())               # Set of all container IDs
        E = [c.id for c in self.containers.values() if c.type == 'E']  # Export containers
        I = [c.id for c in self.containers.values() if c.type == 'I']  # Import containers
        K = list(self.barges.keys()) + ['T']                # Set of barges and 'T' representing trucks
        KB = list(self.barges.keys())                       # Set of barges only

        # Define parameters
        Wc = {c.id: c.size for c in self.containers.values()}  # Wc: Container sizes
        Rc = {c.id: c.release_date for c in self.containers.values() if c.type == 'E'}  # Rc: Release dates for export containers
        Oc = {c.id: c.opening_date for c in self.containers.values()}  # Oc: Opening dates for all containers
        Dc = {c.id: c.closing_date for c in self.containers.values()}  # Dc: Closing dates for all containers

        # Zcj: Indicator if container c is associated with node j
        Zcj = {}
        for c in self.containers.values():
            if c.type == 'E':
                Zcj[c.id, c.destination] = 1  # Export containers are associated with their destination
            else:
                Zcj[c.id, c.origin] = 1      # Import containers are associated with their origin

        HBk = {k: self.barges[k].fixed_cost for k in self.barges.keys()}  # HBk: Fixed costs for each barge
        Qk = {k: self.barges[k].capacity for k in self.barges.keys()}     # Qk: Capacities for each barge
        Tij = {(arc[0], arc[1]): arc[2] for arc in arcs}  # Tij: Travel times between nodes

        # Handling time per container (minutes)
        L = 0.5
        gamma = 50   # Penalty cost for visiting sea terminals unnecessarily

        # Define variables

        # f_ck: Binary variable indicating if container c is assigned to vehicle k
        f_ck = {}
        for c in C:
            for k in K:
                f_ck[c, k] = model.addVar(vtype=GRB.BINARY, name=f"f_{c}_{k}")

        # x_ijk: Binary variable indicating if barge k traverses arc (i, j)
        x_ijk = {}
        for k in KB:
            x_ijk[k] = {}
            for arc in arcs:
                i, j, _ = arc
                x_ijk[k][(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        # p_jk: Continuous variable representing import quantities loaded by barge k at terminal j
        # d_jk: Continuous variable representing export quantities unloaded by barge k at terminal j
        p_jk = {}
        d_jk = {}
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    p_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{j}_{k}")
                    d_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_{j}_{k}")

        # y_ijk: Continuous variable for import containers on arc (i, j) by barge k
        # z_ijk: Continuous variable for export containers on arc (i, j) by barge k
        y_ijk = {}
        z_ijk = {}
        for k in KB:
            y_ijk[k] = {}
            z_ijk[k] = {}
            for arc in arcs:
                i, j, _ = arc
                y_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}_{j}_{k}")
                z_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}_{k}")

        # t_jk: Continuous variable representing the arrival time of barge k at node j
        t_jk = {}
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal' or self.nodes[j].type == 'depot':
                    t_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{j}_{k}")

        # Set the objective function
        # Minimize total cost: truck costs, barge fixed costs, barge travel time costs, penalties
        model.setObjective(
            quicksum(
                f_ck[c, 'T'] * self.trucks[k].cost_per_container
                for c in C
                for k in self.trucks.keys()
                if (c, k) in f_ck and f_ck[c, k]
            ) +  # Truck costs
            quicksum(
                x_ijk[k][(i, j)] * HBk[k]
                for k in KB for (i, j) in x_ijk[k]
            ) +  # Barge fixed costs
            quicksum(
                Tij[i, j] * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k]
            ) +  # Barge travel time costs
            quicksum(
                gamma * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k] if self.nodes[j].type == 'terminal'
            ),  # Penalty for visiting sea terminals unnecessarily
            GRB.MINIMIZE
        )

        # Define Constraints

        # (1) Each container is allocated to exactly one barge or truck
        for c in C:
            model.addConstr(
                quicksum(f_ck[c, k] for k in K) == 1,
                name=f"Assignment_{c}"
            )

        # (2) Flow conservation for x_ijk (Barge Routes)
        for k in KB:
            # Identify depots
            depots = [node.id for node in self.nodes.values() if node.type == 'depot']
            for depot in depots:
                # Each barge can depart from a depot to at most one outgoing arc
                model.addConstr(
                    quicksum(x_ijk[k][(depot, j)] for j in N if j != depot and (depot, j) in x_ijk[k]) <= 1,
                    name=f"Depart_{k}_{depot}"
                )
            for i in N:
                if i not in depots:
                    # For non-depot nodes, ensure flow conservation
                    model.addConstr(
                        quicksum(x_ijk[k][(i, j)] for j in N if j != i and (i, j) in x_ijk[k]) -
                        quicksum(x_ijk[k][(j, i)] for j in N if j != i and (j, i) in x_ijk[k]) == 0,
                        name=f"Flow_{i}_{k}"
                    )

        # (3) Import quantities loaded by barge k at sea terminal j
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    model.addConstr(
                        p_jk[j, k] == quicksum(
                            Wc[c] * Zcj.get((c, j), 0) * f_ck[c, k]
                            for c in I
                        ),
                        name=f"Import_{j}_{k}"
                    )

        # (4) Export quantities unloaded by barge k at sea terminal j
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    model.addConstr(
                        d_jk[j, k] == quicksum(
                            Wc[c] * Zcj.get((c, j), 0) * f_ck[c, k]
                            for c in E
                        ),
                        name=f"Export_{j}_{k}"
                    )

        # (5) Flow equations for y_ijk (import containers)
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    inflow = quicksum(y_ijk[k][(j, i)] for i in N if i != j and (j, i) in y_ijk[k])
                    outflow = quicksum(y_ijk[k][(i, j)] for i in N if i != j and (i, j) in y_ijk[k])
                    model.addConstr(
                        inflow - outflow == p_jk[j, k],
                        name=f"ImportFlow_{j}_{k}"
                    )

        # (6) Flow equations for z_ijk (export containers)
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    inflow = quicksum(z_ijk[k][(i, j)] for i in N if i != j and (i, j) in z_ijk[k])
                    outflow = quicksum(z_ijk[k][(j, i)] for i in N if i != j and (j, i) in z_ijk[k])
                    model.addConstr(
                        inflow - outflow == d_jk[j, k],
                        name=f"ExportFlow_{j}_{k}"
                    )

        # (7) Capacity constraints for barges on each arc
        for k in KB:
            for arc in arcs:
                i, j, _ = arc
                if (i, j) in x_ijk[k]:
                    model.addConstr(
                        y_ijk[k][(i, j)] + z_ijk[k][(i, j)] <= Qk[k] * x_ijk[k][(i, j)],
                        name=f"Capacity_{i}_{j}_{k}"
                    )

        # (8) Barge departure time after release of export containers
        for c in E:
            for k in KB:
                if c in Rc:
                    depot = self.containers[c].origin
                    model.addConstr(
                        t_jk[depot, k] >= Rc[c] * f_ck[c, k],
                        name=f"BargeDeparture_{c}_{k}"
                    )

        # (9) Time calculation constraints linking arrival times at consecutive nodes
        for k in KB:
            for arc in arcs:
                i, j, _ = arc
                handling_time = L * quicksum(
                    Zcj.get((c, i), 0) * f_ck[c, k] for c in C
                )
                if (i, j) in x_ijk[k]:
                    model.addConstr(
                        t_jk[j, k] >= t_jk[i, k] + handling_time + Tij[i, j] - (1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeLower_{i}_{j}_{k}"
                    )
                    model.addConstr(
                        t_jk[j, k] <= t_jk[i, k] + handling_time + Tij[i, j] + (1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeUpper_{i}_{j}_{k}"
                    )

        # (10) Time windows at sea terminals for container operations
        for c in C:
            for k in KB:
                for j in N:
                    if self.nodes[j].type == 'terminal' and Zcj.get((c, j), 0) == 1:
                        # Opening time constraint
                        model.addConstr(
                            t_jk[j, k] >= Oc[c] - (1 - f_ck[c, k]) * M,
                            name=f"TimeWindowOpen_{c}_{j}_{k}"
                        )
                        # Closing time constraint
                        model.addConstr(
                            t_jk[j, k] <= Dc[c] + (1 - f_ck[c, k]) * M,
                            name=f"TimeWindowClose_{c}_{j}_{k}"
                        )

        # Update the model with all variables and constraints
        model.update()

        # Set Gurobi parameters
        model.setParam('OutputFlag', False)    # Disable solver output
        model.setParam('TimeLimit', 300)       # Set a time limit of 5 minutes (300 seconds)

        # Start the optimization process
        model.optimize()

        # Check the status of the model to ensure feasibility and optimality
        if model.Status != GRB.OPTIMAL and model.Status != GRB.TIME_LIMIT:
            if model.Status == GRB.UNBOUNDED:
                messagebox.showerror("Optimization Error", "The model cannot be solved because it is unbounded.")
            elif model.Status == GRB.INFEASIBLE:
                messagebox.showerror("Optimization Error", "The model is infeasible; please check your constraints.")
            else:
                messagebox.showerror("Optimization Error", f"Optimization was stopped with status {model.Status}.")
            return None

        return model

    def display_results(self, model, variables):
        """
        Displays the optimization results in a new window.
        Args:
            model (gurobipy.Model): The optimized Gurobi model.
            variables (dict): Dictionary containing variable values.
        """
        results_window = tk.Toplevel(self.master)
        results_window.title("Optimization Results")

        # Tabs for Objective, Container Allocations, Barge Routes, Truck Routes
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Objective Value Tab
        objective_frame = ttk.Frame(notebook)
        notebook.add(objective_frame, text="Objective Value")
        objective_text = tk.Text(objective_frame, width=80, height=5)
        objective_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        objective_scroll = tk.Scrollbar(objective_frame, command=objective_text.yview)
        objective_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        objective_text.config(yscrollcommand=objective_scroll.set)
        objective_text.insert(tk.END, f"Optimal Objective Value: {model.ObjVal}\n")

        # Container Allocations Tab
        allocations_frame = ttk.Frame(notebook)
        notebook.add(allocations_frame, text="Container Allocations")
        allocations_text = tk.Text(allocations_frame, width=80, height=20)
        allocations_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        allocations_scroll = tk.Scrollbar(allocations_frame, command=allocations_text.yview)
        allocations_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        allocations_text.config(yscrollcommand=allocations_scroll.set)
        allocations_text.insert(tk.END, "Container Allocations:\n")
        for (c, k), value in variables['f_ck'].items():
            if value > 0.5:
                if k == 'T':
                    allocations_text.insert(tk.END, f"Container {c} is allocated to Truck\n")
                else:
                    allocations_text.insert(tk.END, f"Container {c} is allocated to Barge {k}\n")

        # Barge Routes Tab
        routes_frame = ttk.Frame(notebook)
        notebook.add(routes_frame, text="Barge Routes")
        routes_text = tk.Text(routes_frame, width=80, height=20)
        routes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        routes_scroll = tk.Scrollbar(routes_frame, command=routes_text.yview)
        routes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        routes_text.config(yscrollcommand=routes_scroll.set)
        routes_text.insert(tk.END, "Barge Routes:\n")
        for k, arcs in variables['x_ijk'].items():
            routes_text.insert(tk.END, f"\nBarge {k} Route:\n")
            route = []
            for (i, j), val in arcs.items():
                if val > 0.5:
                    route.append((i, j))
            if route:
                for arc in route:
                    routes_text.insert(tk.END, f"{arc[0]} -> {arc[1]}\n")
            else:
                routes_text.insert(tk.END, "No route assigned.\n")

        # Truck Routes Tab
        truck_routes_frame = ttk.Frame(notebook)
        notebook.add(truck_routes_frame, text="Truck Routes")
        truck_routes_text = tk.Text(truck_routes_frame, width=80, height=20)
        truck_routes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        truck_routes_scroll = tk.Scrollbar(truck_routes_frame, command=truck_routes_text.yview)
        truck_routes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        truck_routes_text.config(yscrollcommand=truck_routes_scroll.set)
        truck_routes_text.insert(tk.END, "Truck Routes:\n")
        for k in self.trucks.keys():
            # Find containers allocated to this truck
            allocated_containers = [c for (c, veh), val in variables['f_ck'].items() if veh == k and val > 0.5]
            if allocated_containers:
                # Determine unique origins and destinations
                origins = set(self.containers[c].origin for c in allocated_containers)
                destinations = set(self.containers[c].destination for c in allocated_containers)
                # For simplicity, assume one origin and one destination per truck
                if len(origins) == 1 and len(destinations) == 1:
                    origin = list(origins)[0]
                    destination = list(destinations)[0]
                    truck_routes_text.insert(tk.END, f"\nTruck {k} Route:\n")
                    truck_routes_text.insert(tk.END, f"{origin} -> {destination}\n")
                else:
                    # Handle multiple origins/destinations or implement routing logic
                    truck_routes_text.insert(tk.END, f"\nTruck {k} Route:\n")
                    for c in allocated_containers:
                        origin = self.containers[c].origin
                        destination = self.containers[c].destination
                        truck_routes_text.insert(tk.END, f"{origin} -> {destination}\n")
            else:
                truck_routes_text.insert(tk.END, f"\nTruck {k} Route:\nNo route assigned.\n")

    def visualize_routes(self, save_map_path, variables):
        """
        Visualizes the barge and truck routes on the canvas.
        Args:
            save_map_path (str or None): Path to save the HTML map (if needed). Not used in this implementation.
            variables (dict): Dictionary containing variable values.
        """
        # Remove existing route lines and truck routes
        self.canvas.delete("route")
        self.canvas.delete("truck_route")

        # Visualize Barge Routes
        for k, arcs in variables['x_ijk'].items():
            route_points = []
            for (i, j), val in arcs.items():
                if val > 0.5:
                    origin = self.nodes[i]
                    destination = self.nodes[j]
                    route_points.append((origin.x, origin.y))
                    route_points.append((destination.x, destination.y))
                    # Draw a line segment for this arc
                    self.canvas.create_line(origin.x, origin.y, destination.x, destination.y, fill='red', width=2, tags="route")
            if route_points:
                # Mark the start and end points
                start_node = self.nodes[route_points[0][0]]
                end_node = self.nodes[route_points[-1][0]]
                self.canvas.create_oval(start_node.x - 7, start_node.y - 7, start_node.x + 7, start_node.y + 7, outline='black', width=2, tags="route")
                self.canvas.create_text(start_node.x, start_node.y - 10, text=f"Barge {k} Start", fill='black', font=('Arial', 10, 'bold'), tags="route")
                self.canvas.create_oval(end_node.x - 7, end_node.y - 7, end_node.x + 7, end_node.y + 7, outline='black', width=2, tags="route")
                self.canvas.create_text(end_node.x, end_node.y - 10, text=f"Barge {k} End", fill='black', font=('Arial', 10, 'bold'), tags="route")

        # Visualize Truck Routes
        # Iterate through trucks and find their assignments
        for k in self.trucks.keys():
            # Find containers allocated to this truck
            allocated_containers = [c for (c, veh), val in variables['f_ck'].items() if veh == k and val > 0.5]
            if allocated_containers:
                # Determine unique origins and destinations
                origins = set(self.containers[c].origin for c in allocated_containers)
                destinations = set(self.containers[c].destination for c in allocated_containers)
                # For simplicity, connect each origin to its destination
                for c in allocated_containers:
                    origin = self.nodes[self.containers[c].origin]
                    destination = self.nodes[self.containers[c].destination]
                    # Draw a dashed blue line for truck routes
                    self.canvas.create_line(origin.x, origin.y, destination.x, destination.y, fill='blue', width=2, dash=(4, 2), tags="truck_route")
                    # Mark the start and end points
                    self.canvas.create_oval(origin.x - 7, origin.y - 7, origin.x + 7, origin.y + 7, outline='black', width=2, tags="truck_route")
                    self.canvas.create_text(origin.x, origin.y - 10, text=f"Truck {k} Start", fill='black', font=('Arial', 10, 'bold'), tags="truck_route")
                    self.canvas.create_oval(destination.x - 7, destination.y - 7, destination.x + 7, destination.y + 7, outline='black', width=2, tags="truck_route")
                    self.canvas.create_text(destination.x, destination.y - 10, text=f"Truck {k} End", fill='black', font=('Arial', 10, 'bold'), tags="truck_route")

        # Notify the user
        messagebox.showinfo("Routes Visualized", "Barge and Truck routes have been visualized on the map.")

    def construct_arcs(self):
        """
        Constructs the arcs based on nodes' pixel coordinates.
        Returns:
            nodes (dict): Dictionary of Node objects keyed by node ID.
            arcs (list): List of Arc tuples representing possible routes (origin, destination, travel_time).
        """
        nodes = self.nodes
        arcs = []

        # Calculate travel times between nodes (in minutes)
        # Assume average speed of 20 pixels/minute for barges
        average_speed_pixels_per_min = 20
        for i in nodes:
            for j in nodes:
                if i != j:
                    origin = nodes[i]
                    destination = nodes[j]
                    distance = sqrt((origin.x - destination.x)**2 + (origin.y - destination.y)**2)  # pixels
                    travel_time = distance / average_speed_pixels_per_min  # minutes
                    arcs.append((i, j, travel_time))
        return nodes, arcs

    def build_and_solve_model(self, nodes, arcs):
        """
        Builds and solves the optimization model using Gurobi.
        Args:
            nodes (dict): Dictionary of Node objects keyed by node ID.
            arcs (list): List of Arc tuples representing possible routes (origin, destination, travel_time).
        Returns:
            model (gurobipy.Model): The optimized Gurobi model, or None if infeasible.
        """
        model = Model("TransportationOptimization")

        # Big M
        M = 10000  # A large constant used in Big M method for conditional constraints

        # Define sets
        N = list(nodes.keys())                         # Set of all node IDs
        C = list(self.containers.keys())               # Set of all container IDs
        E = [c.id for c in self.containers.values() if c.type == 'E']  # Export containers
        I = [c.id for c in self.containers.values() if c.type == 'I']  # Import containers
        K = list(self.barges.keys()) + ['T']                # Set of barges and 'T' representing trucks
        KB = list(self.barges.keys())                       # Set of barges only

        # Define parameters
        Wc = {c.id: c.size for c in self.containers.values()}  # Wc: Container sizes
        Rc = {c.id: c.release_date for c in self.containers.values() if c.type == 'E'}  # Rc: Release dates for export containers
        Oc = {c.id: c.opening_date for c in self.containers.values()}  # Oc: Opening dates for all containers
        Dc = {c.id: c.closing_date for c in self.containers.values()}  # Dc: Closing dates for all containers

        # Zcj: Indicator if container c is associated with node j
        Zcj = {}
        for c in self.containers.values():
            if c.type == 'E':
                Zcj[c.id, c.destination] = 1  # Export containers are associated with their destination
            else:
                Zcj[c.id, c.origin] = 1      # Import containers are associated with their origin

        HBk = {k: self.barges[k].fixed_cost for k in self.barges.keys()}  # HBk: Fixed costs for each barge
        Qk = {k: self.barges[k].capacity for k in self.barges.keys()}     # Qk: Capacities for each barge
        Tij = {(arc[0], arc[1]): arc[2] for arc in arcs}  # Tij: Travel times between nodes

        # Handling time per container (minutes)
        L = 0.5
        gamma = 50   # Penalty cost for visiting sea terminals unnecessarily

        # Define variables

        # f_ck: Binary variable indicating if container c is assigned to vehicle k
        f_ck = {}
        for c in C:
            for k in K:
                f_ck[c, k] = model.addVar(vtype=GRB.BINARY, name=f"f_{c}_{k}")

        # x_ijk: Binary variable indicating if barge k traverses arc (i, j)
        x_ijk = {}
        for k in KB:
            x_ijk[k] = {}
            for arc in arcs:
                i, j, _ = arc
                x_ijk[k][(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        # p_jk: Continuous variable representing import quantities loaded by barge k at terminal j
        # d_jk: Continuous variable representing export quantities unloaded by barge k at terminal j
        p_jk = {}
        d_jk = {}
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    p_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{j}_{k}")
                    d_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_{j}_{k}")

        # y_ijk: Continuous variable for import containers on arc (i, j) by barge k
        # z_ijk: Continuous variable for export containers on arc (i, j) by barge k
        y_ijk = {}
        z_ijk = {}
        for k in KB:
            y_ijk[k] = {}
            z_ijk[k] = {}
            for arc in arcs:
                i, j, _ = arc
                y_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}_{j}_{k}")
                z_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}_{k}")

        # t_jk: Continuous variable representing the arrival time of barge k at node j
        t_jk = {}
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal' or self.nodes[j].type == 'depot':
                    t_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{j}_{k}")

        # Set the objective function
        # Minimize total cost: truck costs, barge fixed costs, barge travel time costs, penalties
        model.setObjective(
            quicksum(
                f_ck[c, 'T'] * self.trucks[k].cost_per_container
                for c in C
                for k in self.trucks.keys()
                if (c, k) in f_ck and f_ck[c, k]
            ) +  # Truck costs
            quicksum(
                x_ijk[k][(i, j)] * HBk[k]
                for k in KB for (i, j) in x_ijk[k]
            ) +  # Barge fixed costs
            quicksum(
                Tij[i, j] * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k]
            ) +  # Barge travel time costs
            quicksum(
                gamma * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k] if self.nodes[j].type == 'terminal'
            ),  # Penalty for visiting sea terminals unnecessarily
            GRB.MINIMIZE
        )

        # Define Constraints

        # (1) Each container is allocated to exactly one barge or truck
        for c in C:
            model.addConstr(
                quicksum(f_ck[c, k] for k in K) == 1,
                name=f"Assignment_{c}"
            )

        # (2) Flow conservation for x_ijk (Barge Routes)
        for k in KB:
            # Identify depots
            depots = [node.id for node in self.nodes.values() if node.type == 'depot']
            for depot in depots:
                # Each barge can depart from a depot to at most one outgoing arc
                model.addConstr(
                    quicksum(x_ijk[k][(depot, j)] for j in N if j != depot and (depot, j) in x_ijk[k]) <= 1,
                    name=f"Depart_{k}_{depot}"
                )
            for i in N:
                if i not in depots:
                    # For non-depot nodes, ensure flow conservation
                    model.addConstr(
                        quicksum(x_ijk[k][(i, j)] for j in N if j != i and (i, j) in x_ijk[k]) -
                        quicksum(x_ijk[k][(j, i)] for j in N if j != i and (j, i) in x_ijk[k]) == 0,
                        name=f"Flow_{i}_{k}"
                    )

        # (3) Import quantities loaded by barge k at sea terminal j
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    model.addConstr(
                        p_jk[j, k] == quicksum(
                            Wc[c] * Zcj.get((c, j), 0) * f_ck[c, k]
                            for c in I
                        ),
                        name=f"Import_{j}_{k}"
                    )

        # (4) Export quantities unloaded by barge k at sea terminal j
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    model.addConstr(
                        d_jk[j, k] == quicksum(
                            Wc[c] * Zcj.get((c, j), 0) * f_ck[c, k]
                            for c in E
                        ),
                        name=f"Export_{j}_{k}"
                    )

        # (5) Flow equations for y_ijk (import containers)
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    inflow = quicksum(y_ijk[k][(j, i)] for i in N if i != j and (j, i) in y_ijk[k])
                    outflow = quicksum(y_ijk[k][(i, j)] for i in N if i != j and (i, j) in y_ijk[k])
                    model.addConstr(
                        inflow - outflow == p_jk[j, k],
                        name=f"ImportFlow_{j}_{k}"
                    )

        # (6) Flow equations for z_ijk (export containers)
        for k in KB:
            for j in N:
                if self.nodes[j].type == 'terminal':
                    inflow = quicksum(z_ijk[k][(i, j)] for i in N if i != j and (i, j) in z_ijk[k])
                    outflow = quicksum(z_ijk[k][(j, i)] for i in N if i != j and (j, i) in z_ijk[k])
                    model.addConstr(
                        inflow - outflow == d_jk[j, k],
                        name=f"ExportFlow_{j}_{k}"
                    )

        # (7) Capacity constraints for barges on each arc
        for k in KB:
            for arc in arcs:
                i, j, _ = arc
                if (i, j) in x_ijk[k]:
                    model.addConstr(
                        y_ijk[k][(i, j)] + z_ijk[k][(i, j)] <= Qk[k] * x_ijk[k][(i, j)],
                        name=f"Capacity_{i}_{j}_{k}"
                    )

        # (8) Barge departure time after release of export containers
        for c in E:
            for k in KB:
                if c in Rc:
                    depot = self.containers[c].origin
                    model.addConstr(
                        t_jk[depot, k] >= Rc[c] * f_ck[c, k],
                        name=f"BargeDeparture_{c}_{k}"
                    )

        # (9) Time calculation constraints linking arrival times at consecutive nodes
        for k in KB:
            for arc in arcs:
                i, j, _ = arc
                handling_time = L * quicksum(
                    Zcj.get((c, i), 0) * f_ck[c, k] for c in C
                )
                if (i, j) in x_ijk[k]:
                    model.addConstr(
                        t_jk[j, k] >= t_jk[i, k] + handling_time + Tij[i, j] - (1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeLower_{i}_{j}_{k}"
                    )
                    model.addConstr(
                        t_jk[j, k] <= t_jk[i, k] + handling_time + Tij[i, j] + (1 - x_ijk[k][(i, j)]) * M,
                        name=f"TimeUpper_{i}_{j}_{k}"
                    )

        # (10) Time windows at sea terminals for container operations
        for c in C:
            for k in KB:
                for j in N:
                    if self.nodes[j].type == 'terminal' and Zcj.get((c, j), 0) == 1:
                        # Opening time constraint
                        model.addConstr(
                            t_jk[j, k] >= Oc[c] - (1 - f_ck[c, k]) * M,
                            name=f"TimeWindowOpen_{c}_{j}_{k}"
                        )
                        # Closing time constraint
                        model.addConstr(
                            t_jk[j, k] <= Dc[c] + (1 - f_ck[c, k]) * M,
                            name=f"TimeWindowClose_{c}_{j}_{k}"
                        )

        # Update the model with all variables and constraints
        model.update()

        # Set Gurobi parameters
        model.setParam('OutputFlag', False)    # Disable solver output
        model.setParam('TimeLimit', 300)       # Set a time limit of 5 minutes (300 seconds)

        # Start the optimization process
        model.optimize()

        # Check the status of the model to ensure feasibility and optimality
        if model.Status != GRB.OPTIMAL and model.Status != GRB.TIME_LIMIT:
            if model.Status == GRB.UNBOUNDED:
                messagebox.showerror("Optimization Error", "The model cannot be solved because it is unbounded.")
            elif model.Status == GRB.INFEASIBLE:
                messagebox.showerror("Optimization Error", "The model is infeasible; please check your constraints.")
            else:
                messagebox.showerror("Optimization Error", f"Optimization was stopped with status {model.Status}.")
            return None

        return model

    def extract_variables(self, model, nodes, arcs):
        """
        Extracts the necessary variables from the optimized model.
        Args:
            model (gurobipy.Model): The optimized Gurobi model.
            nodes (dict): Dictionary of Node objects.
            arcs (list): List of Arc tuples.
        Returns:
            variables (dict): Dictionary containing variable values.
        """
        variables = {}

        # Extract f_ck variables
        f_ck = {}
        for v in model.getVars():
            if v.varName.startswith("f_"):
                parts = v.varName.split("_")
                c = int(parts[1])
                k = parts[2]
                f_ck[c, k] = v.X
        variables['f_ck'] = f_ck

        # Extract x_ijk variables
        x_ijk = {}
        for v in model.getVars():
            if v.varName.startswith("x_"):
                parts = v.varName.split("_")
                i = int(parts[1])
                j = int(parts[2])
                k = parts[3]
                if k not in x_ijk:
                    x_ijk[k] = {}
                x_ijk[k][(i, j)] = v.X
        variables['x_ijk'] = x_ijk

        return variables

    def display_results(self, model, variables):
        """
        Displays the optimization results in a new window.
        Args:
            model (gurobipy.Model): The optimized Gurobi model.
            variables (dict): Dictionary containing variable values.
        """
        results_window = tk.Toplevel(self.master)
        results_window.title("Optimization Results")

        # Tabs for Objective, Container Allocations, Barge Routes, Truck Routes
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Objective Value Tab
        objective_frame = ttk.Frame(notebook)
        notebook.add(objective_frame, text="Objective Value")
        objective_text = tk.Text(objective_frame, width=80, height=5)
        objective_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        objective_scroll = tk.Scrollbar(objective_frame, command=objective_text.yview)
        objective_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        objective_text.config(yscrollcommand=objective_scroll.set)
        objective_text.insert(tk.END, f"Optimal Objective Value: {model.ObjVal}\n")

        # Container Allocations Tab
        allocations_frame = ttk.Frame(notebook)
        notebook.add(allocations_frame, text="Container Allocations")
        allocations_text = tk.Text(allocations_frame, width=80, height=20)
        allocations_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        allocations_scroll = tk.Scrollbar(allocations_frame, command=allocations_text.yview)
        allocations_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        allocations_text.config(yscrollcommand=allocations_scroll.set)
        allocations_text.insert(tk.END, "Container Allocations:\n")
        for (c, k), value in variables['f_ck'].items():
            if value > 0.5:
                if k == 'T':
                    allocations_text.insert(tk.END, f"Container {c} is allocated to Truck\n")
                else:
                    allocations_text.insert(tk.END, f"Container {c} is allocated to Barge {k}\n")

        # Barge Routes Tab
        routes_frame = ttk.Frame(notebook)
        notebook.add(routes_frame, text="Barge Routes")
        routes_text = tk.Text(routes_frame, width=80, height=20)
        routes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        routes_scroll = tk.Scrollbar(routes_frame, command=routes_text.yview)
        routes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        routes_text.config(yscrollcommand=routes_scroll.set)
        routes_text.insert(tk.END, "Barge Routes:\n")
        for k, arcs in variables['x_ijk'].items():
            routes_text.insert(tk.END, f"\nBarge {k} Route:\n")
            route = []
            for (i, j), val in arcs.items():
                if val > 0.5:
                    route.append((i, j))
            if route:
                for arc in route:
                    routes_text.insert(tk.END, f"{arc[0]} -> {arc[1]}\n")
            else:
                routes_text.insert(tk.END, "No route assigned.\n")

        # Truck Routes Tab
        truck_routes_frame = ttk.Frame(notebook)
        notebook.add(truck_routes_frame, text="Truck Routes")
        truck_routes_text = tk.Text(truck_routes_frame, width=80, height=20)
        truck_routes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        truck_routes_scroll = tk.Scrollbar(truck_routes_frame, command=truck_routes_text.yview)
        truck_routes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        truck_routes_text.config(yscrollcommand=truck_routes_scroll.set)
        truck_routes_text.insert(tk.END, "Truck Routes:\n")
        for k in self.trucks.keys():
            # Find containers allocated to this truck
            allocated_containers = [c for (c, veh), val in variables['f_ck'].items() if veh == k and val > 0.5]
            if allocated_containers:
                # Determine unique origins and destinations
                origins = set(self.containers[c].origin for c in allocated_containers)
                destinations = set(self.containers[c].destination for c in allocated_containers)
                # For simplicity, assume one origin and one destination per truck
                if len(origins) == 1 and len(destinations) == 1:
                    origin = list(origins)[0]
                    destination = list(destinations)[0]
                    truck_routes_text.insert(tk.END, f"\nTruck {k} Route:\n")
                    truck_routes_text.insert(tk.END, f"{origin} -> {destination}\n")
                else:
                    # Handle multiple origins/destinations or implement routing logic
                    truck_routes_text.insert(tk.END, f"\nTruck {k} Route:\n")
                    for c in allocated_containers:
                        origin = self.containers[c].origin
                        destination = self.containers[c].destination
                        truck_routes_text.insert(tk.END, f"{origin} -> {destination}\n")
            else:
                truck_routes_text.insert(tk.END, f"\nTruck {k} Route:\nNo route assigned.\n")

    def visualize_routes(self, save_map_path, variables):
        """
        Visualizes the barge and truck routes on the canvas.
        Args:
            save_map_path (str or None): Path to save the HTML map (if needed). Not used in this implementation.
            variables (dict): Dictionary containing variable values.
        """
        # Remove existing route lines and truck routes
        self.canvas.delete("route")
        self.canvas.delete("truck_route")

        # Visualize Barge Routes
        for k, arcs in variables['x_ijk'].items():
            route_points = []
            for (i, j), val in arcs.items():
                if val > 0.5:
                    origin = self.nodes[i]
                    destination = self.nodes[j]
                    route_points.append((origin.x, origin.y))
                    route_points.append((destination.x, destination.y))
                    # Draw a line segment for this arc
                    self.canvas.create_line(origin.x, origin.y, destination.x, destination.y, fill='red', width=2, tags="route")
            if route_points:
                # Mark the start and end points
                start_node = self.nodes[route_points[0][0]]
                end_node = self.nodes[route_points[-1][0]]
                self.canvas.create_oval(start_node.x - 7, start_node.y - 7, start_node.x + 7, start_node.y + 7, outline='black', width=2, tags="route")
                self.canvas.create_text(start_node.x, start_node.y - 10, text=f"Barge {k} Start", fill='black', font=('Arial', 10, 'bold'), tags="route")
                self.canvas.create_oval(end_node.x - 7, end_node.y - 7, end_node.x + 7, end_node.y + 7, outline='black', width=2, tags="route")
                self.canvas.create_text(end_node.x, end_node.y - 10, text=f"Barge {k} End", fill='black', font=('Arial', 10, 'bold'), tags="route")

        # Visualize Truck Routes
        # Iterate through trucks and find their assignments
        for k in self.trucks.keys():
            # Find containers allocated to this truck
            allocated_containers = [c for (c, veh), val in variables['f_ck'].items() if veh == k and val > 0.5]
            if allocated_containers:
                # Determine unique origins and destinations
                origins = set(self.containers[c].origin for c in allocated_containers)
                destinations = set(self.containers[c].destination for c in allocated_containers)
                # For simplicity, connect each origin to its destination
                for c in allocated_containers:
                    origin = self.nodes[self.containers[c].origin]
                    destination = self.nodes[self.containers[c].destination]
                    # Draw a dashed blue line for truck routes
                    self.canvas.create_line(origin.x, origin.y, destination.x, destination.y, fill='blue', width=2, dash=(4, 2), tags="truck_route")
                    # Mark the start and end points
                    self.canvas.create_oval(origin.x - 7, origin.y - 7, origin.x + 7, origin.y + 7, outline='black', width=2, tags="truck_route")
                    self.canvas.create_text(origin.x, origin.y - 10, text=f"Truck {k} Start", fill='black', font=('Arial', 10, 'bold'), tags="truck_route")
                    self.canvas.create_oval(destination.x - 7, destination.y - 7, destination.x + 7, destination.y + 7, outline='black', width=2, tags="truck_route")
                    self.canvas.create_text(destination.x, destination.y - 10, text=f"Truck {k} End", fill='black', font=('Arial', 10, 'bold'), tags="truck_route")

        # Notify the user
        messagebox.showinfo("Routes Visualized", "Barge and Truck routes have been visualized on the map.")

    def save_network(self):
        # Prompt user to select a file to save
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not save_path:
            return

        # Prepare data
        data = {
            "nodes": {node_id: {"x": node.x, "y": node.y, "type": node.type}
                      for node_id, node in self.nodes.items()},
            "containers": {container_id: {
                "size": container.size,
                "release_date": container.release_date,
                "opening_date": container.opening_date,
                "closing_date": container.closing_date,
                "origin": container.origin,
                "destination": container.destination,
                "type": container.type
            } for container_id, container in self.containers.items()},
            "barges": {barge_id: {
                "capacity": barge.capacity,
                "fixed_cost": barge.fixed_cost
            } for barge_id, barge in self.barges.items()},
            "trucks": {truck_id: {
                "cost_per_container": truck.cost_per_container
            } for truck_id, truck in self.trucks.items()},
            "next_ids": {
                "node": self.next_node_id,
                "container": self.next_container_id,
                "barge": self.next_barge_id,
                "truck": self.next_truck_id
            }
        }

        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Saved", f"Network data saved to {save_path}.")

    def load_network(self):
        # Prompt user to select a JSON file
        load_path = filedialog.askopenfilename(title="Load Network Data", filetypes=[("JSON Files", "*.json")])
        if not load_path:
            return

        if not os.path.exists(load_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return

        # Load data from JSON
        with open(load_path, 'r') as f:
            data = json.load(f)

        # Clear existing data
        self.clear_data()

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node_id = int(node_id)
            node = Node(node_id, node_data["x"], node_data["y"], node_data["type"])
            self.nodes[node_id] = node
            self.draw_node(node)

        # Load containers
        for container_id, container_data in data.get("containers", {}).items():
            container_id = int(container_id)
            container = Container(container_id, container_data["size"],
                                  container_data["release_date"],
                                  container_data["opening_date"],
                                  container_data["closing_date"],
                                  container_data["origin"],
                                  container_data["destination"],
                                  container_data["type"])
            self.containers[container_id] = container

        # Load barges
        for barge_id, barge_data in data.get("barges", {}).items():
            barge_id = int(barge_id)
            barge = Barge(barge_id, barge_data["capacity"], barge_data["fixed_cost"])
            self.barges[barge_id] = barge

        # Load trucks
        for truck_id, truck_data in data.get("trucks", {}).items():
            truck_id = int(truck_id)
            truck = Truck(truck_id, truck_data["cost_per_container"])
            self.trucks[truck_id] = truck

        # Load next IDs
        self.next_node_id = data.get("next_ids", {}).get("node", self.next_node_id)
        self.next_container_id = data.get("next_ids", {}).get("container", self.next_container_id)
        self.next_barge_id = data.get("next_ids", {}).get("barge", self.next_barge_id)
        self.next_truck_id = data.get("next_ids", {}).get("truck", self.next_truck_id)

        messagebox.showinfo("Loaded", f"Network data loaded from {load_path}.")

    def clear_data(self):
        # Clear all data
        self.nodes.clear()
        self.containers.clear()
        self.barges.clear()
        self.trucks.clear()
        self.next_node_id = 0
        self.next_container_id = 1
        self.next_barge_id = 1
        self.next_truck_id = 1

        # Clear canvas (remove all except the map image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)

    def show_about(self):
        messagebox.showinfo("About", "Interactive Network Creator with Optimizer\nDeveloped with Tkinter and Gurobi.")

    def solve_model(self):
        if not self.nodes:
            messagebox.showerror("No Nodes", "Please add nodes before solving the model.")
            return
        if not self.containers:
            messagebox.showerror("No Containers", "Please add containers before solving the model.")
            return
        if not self.barges:
            messagebox.showerror("No Barges", "Please add barges before solving the model.")
            return
        if not self.trucks:
            messagebox.showerror("No Trucks", "Please add trucks before solving the model.")
            return

        try:
            # Construct the optimization model
            nodes_dict, arcs_list = self.construct_arcs()
            model = self.build_and_solve_model(nodes_dict, arcs_list)
            if model is None:
                return  # Model was infeasible or an error occurred
            # Extract variables
            variables = self.extract_variables(model, nodes_dict, arcs_list)
            # Display results
            self.display_results(model, variables)
            # Visualize routes
            self.visualize_routes(None, variables)  # No need to save as image
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during optimization:\n{str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkCreatorApp(root)
    root.mainloop()
