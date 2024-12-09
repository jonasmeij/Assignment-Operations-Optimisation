
from collections import defaultdict
import threading
import time
import queue
import colorsys  # Import colorsys for color generation
import random  # If needed for other random functionalities
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import json
import os
from math import sqrt
from gurobipy import *
import random  # Added for random container generation
import ast



# Define data classes
class Node:
    def __init__(self, node_id, x, y, node_type='terminal'):
        self.id = node_id
        self.x = x  # Pixel x-coordinate on the map
        self.y = y  # Pixel y-coordinate on the map
        self.type = node_type  # 'depot', 'dummy', or 'terminal'

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
    def __init__(self, barge_id, capacity, fixed_cost, origin):
        self.id = barge_id
        self.capacity = capacity
        self.fixed_cost = fixed_cost
        self.origin = origin  # Origin depot node ID

class Truck:
    def __init__(self, cost_per_container):
        self.cost_per_container = cost_per_container

class Arc:
    def __init__(self, origin, destination, travel_time):
        self.origin = origin                  # Origin node ID
        self.destination = destination        # Destination node ID
        self.travel_time = travel_time        # Tij: Travel time in minutes between origin and destination


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
        self.depot_to_dummy = {}
        self.next_node_id = 0
        self.next_container_id = 1
        self.next_barge_id = 1
        self.next_truck_id = 11  # Start from 11 since trucks are predefined with IDs 9 and 10

        # Predefine trucks
        self.predefine_trucks()
        # Define a list of distinct colors
        self.color_list = [
            'red', 'green', 'blue', 'orange', 'purple',
            'cyan', 'magenta', 'yellow', 'brown', 'pink',
            'lime', 'teal', 'lavender', 'maroon', 'navy',
            'olive', 'coral', 'grey', 'black', 'gold'
        ]

        # Initialize a dictionary to map barge IDs to colors
        self.barge_colors = {}


        # Create the main frames
        self.create_main_frames()

        # Load map
        self.load_map()

        # Create menus and buttons
        self.create_menus()
        self.create_buttons()

        # Bind click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)


    def create_main_frames(self):
        """
        Creates the main frames for the canvas and the button panel.
        """
        # Frame for the map and its scrollbars
        self.map_frame = tk.Frame(self.master)
        self.map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        self.v_scrollbar = tk.Scrollbar(self.map_frame, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        self.h_scrollbar = tk.Scrollbar(self.map_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(
            self.map_frame,
            bg="white",
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set,
            scrollregion=(0, 0, 1000, 1000)  # Initial scroll region; will be updated when image is loaded
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)

        # Frame for buttons on the right
        self.button_frame = tk.Frame(self.master, padx=10, pady=10)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)


    def predefine_trucks(self):
        # Predefine Truck 1
        truck1 = Truck(cost_per_container=150)  # Example values
        self.trucks[9] = truck1

        # Predefine Truck 2
        truck2 = Truck(cost_per_container=200)  # Example values
        self.trucks[10] = truck2

        messagebox.showinfo("Predefined Trucks", "Two trucks (ID 9 and ID 10) have been initialized.")

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

        # Update the scroll region to the size of the image
        self.canvas.config(scrollregion=(0, 0, self.map_photo.width(), self.map_photo.height()))

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

        # Button to generate random containers
        generate_random_containers_btn = tk.Button(button_frame, text="Generate Random Containers", command=self.generate_random_containers)
        generate_random_containers_btn.pack(pady=5, fill=tk.X)  # New button

        # Button to add barge
        add_barge_btn = tk.Button(button_frame, text="Add Barge", command=self.add_barge)
        add_barge_btn.pack(pady=5, fill=tk.X)

        # Remove or disable the "Add Truck" button since trucks are predefined
        # Option 1: Remove the button
        # pass  # No button added for adding trucks

        # Button to view data
        view_data_btn = tk.Button(button_frame, text="View Data", command=self.view_data)
        view_data_btn.pack(pady=5, fill=tk.X)

        # Button to solve the optimization model
        solve_model_btn = tk.Button(button_frame, text="Solve Model", command=self.solve_model)
        solve_model_btn.pack(pady=20, fill=tk.X)

    def on_canvas_click(self, event):
        # Get the absolute coordinates accounting for scrollbars
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Continue with adding the node using the absolute coordinates
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

        if node_type == 'depot':
            # Automatically create a dummy depot
            dummy_node_id = self.next_node_id
            self.next_node_id += 1
            # Position the dummy depot slightly offset from the real depot for visibility
            dummy_x = x + 10  # Offset by 20 pixels; adjust as needed
            dummy_y = y + 10
            dummy_node = Node(dummy_node_id, dummy_x, dummy_y, 'depot_arr')
            self.nodes[dummy_node_id] = dummy_node
            self.depot_to_dummy[node_id] = dummy_node_id
            self.draw_node(dummy_node)
            messagebox.showinfo("Dummy Depot Added", f"Dummy Depot {dummy_node_id} added for Depot {node_id}.")

    def draw_node(self, node):
        # Define colors based on node type
        if node.type == 'depot':
            color = 'blue'
        elif node.type == 'depot_arr':
            color = 'orange'  # Different color for dummy depots
        else:
            color = 'green'
        radius = 5
        self.canvas.create_oval(node.x - radius, node.y - radius, node.x + radius, node.y + radius, fill=color, outline='black')
        self.canvas.create_text(node.x, node.y - 10, text=str(node.id), fill='black', font=('Arial', 10, 'bold'))

    def get_node_ids_by_type(self, node_type):
        """
        Returns a list of node IDs filtered by the specified node type.
        Args:
            node_type (str): 'depot', 'dummy', or 'terminal'
        Returns:
            List of node IDs matching the type.
        """
        return [str(node.id) for node in self.nodes.values() if node.type == node_type]

    import tkinter as tk
    from tkinter import messagebox
    import ast
    import json

    def add_container(self):
        if not any(self.nodes.values()):
            messagebox.showerror("No Nodes", "Please add nodes before adding containers.")
            return

        # Create a new window for container input
        container_window = tk.Toplevel(self.master)
        container_window.title("Add Containers")
        container_window.geometry("900x600")  # Adjust size as needed

        # Create a main frame to hold canvas and scrollbar
        main_frame = tk.Frame(container_window)
        main_frame.pack(fill='both', expand=True)

        # Create a canvas and a vertical scrollbar for the container inputs
        canvas = tk.Canvas(main_frame, borderwidth=0, background="#f0f0f0")
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
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
        headers = ["Size (2=20ft, 4=40ft)", "Release Date", "Opening Date", "Closing Date",
                   "Origin Node ID", "Destination Node ID", "Type", "Action"]
        for col, header in enumerate(headers):
            tk.Label(scrollable_frame, text=header, borderwidth=1, relief="solid", bg="#d3d3d3").grid(row=0, column=col,
                                                                                                      padx=1, pady=1,
                                                                                                      sticky='nsew')

        # Function to add a new container input row
        def add_container_row(data=None):
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
            origin_menu = tk.OptionMenu(scrollable_frame, origin_var,
                                        *self.get_node_ids_by_type('depot') + self.get_node_ids_by_type('depot_arr'))
            origin_menu.grid(row=row, column=4, padx=1, pady=1, sticky='nsew')

            # Destination node
            destination_var = tk.StringVar(scrollable_frame)
            destination_menu = tk.OptionMenu(scrollable_frame, destination_var,
                                             *self.get_node_ids_by_type('terminal') + self.get_node_ids_by_type(
                                                 'depot_arr'))
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
                    # Export: Origin = depot or dummy, Destination = terminal
                    origin_menu['menu'].delete(0, 'end')
                    for depot_id in self.get_node_ids_by_type('depot') + self.get_node_ids_by_type('depot_arr'):
                        origin_menu['menu'].add_command(label=depot_id,
                                                        command=lambda value=depot_id: origin_var.set(value))
                    destination_menu['menu'].delete(0, 'end')
                    for terminal_id in self.get_node_ids_by_type('terminal'):
                        destination_menu['menu'].add_command(label=terminal_id,
                                                             command=lambda value=terminal_id: destination_var.set(
                                                                 value))
                elif container_type == 'I':
                    # Import: Origin = terminal, Destination = dummy depots
                    origin_menu['menu'].delete(0, 'end')
                    for terminal_id in self.get_node_ids_by_type('terminal'):
                        origin_menu['menu'].add_command(label=terminal_id,
                                                        command=lambda value=terminal_id: origin_var.set(value))
                    destination_menu['menu'].delete(0, 'end')
                    for dummy_id in self.get_node_ids_by_type('depot_arr'):
                        destination_menu['menu'].add_command(label=dummy_id,
                                                             command=lambda value=dummy_id: destination_var.set(value))

            # Bind the type_var to the on_type_change function
            type_var.trace('w', on_type_change)

            # Initialize menus based on default type 'E'
            on_type_change()

            # If data is provided, populate the fields
            if data:
                try:
                    size_entry.insert(0, data['size'])
                    release_entry.insert(0, data['release_date'])
                    opening_entry.insert(0, data['opening_date'])
                    closing_entry.insert(0, data['closing_date'])
                    origin_var.set(str(data['origin_node_id']))  # Ensure it's a string for the GUI
                    destination_var.set(str(data['destination_node_id']))  # Ensure it's a string for the GUI
                    type_var.set(data['type'])
                    on_type_change()
                except KeyError as e:
                    messagebox.showerror("Data Error", f"Missing key in data: {e}")

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
            # Remove the widgets from the grid
            for widget in scrollable_frame.grid_slaves(row=row):
                widget.grid_forget()
            # Optionally, remove the entry from container_entries
            # This requires tracking row numbers more carefully
            # For simplicity, we're not implementing it here

        # Function to handle bulk addition from list of tuples
        def add_containers_from_list():
            input_window = tk.Toplevel(container_window)
            input_window.title("Add Containers from List")
            input_window.geometry("500x400")

            tk.Label(input_window, text="Enter list of container tuples:", wraplength=480).pack(pady=10)

            text_input = tk.Text(input_window, wrap='word')
            text_input.pack(padx=10, pady=10, fill='both', expand=True)


            def submit_list():
                input_text = text_input.get("1.0", tk.END).strip()
                if not input_text:
                    messagebox.showerror("Input Error", "Please enter a list of tuples.")
                    return
                try:
                    # Safely evaluate the list of tuples
                    data_list = ast.literal_eval(input_text)
                    if not isinstance(data_list, list):
                        raise ValueError("Input must be a list of tuples.")
                    for item in data_list:
                        if not isinstance(item, tuple) or len(item) != 7:
                            raise ValueError(
                                "Each item must be a tuple with 7 elements: (size, release_date, opening_date, closing_date, origin_node_id, destination_node_id, type)")
                        size, release_date, opening_date, closing_date, origin_node_id, destination_node_id, container_type = item
                        # Validate container_type
                        if container_type not in ('E', 'I'):
                            raise ValueError("Container type must be 'E' or 'I'.")
                        # Prepare data dictionary
                        data = {
                            'size': size,
                            'release_date': release_date,
                            'opening_date': opening_date,
                            'closing_date': closing_date,
                            'origin_node_id': origin_node_id,
                            'destination_node_id': destination_node_id,
                            'type': container_type
                        }
                        add_container_row(data)
                    messagebox.showinfo("Success", "Containers added successfully.")
                    input_window.destroy()
                except Exception as e:
                    messagebox.showerror("Parsing Error", f"An error occurred while parsing the input:\n{e}")

            submit_btn = tk.Button(input_window, text="Add Containers", command=submit_list)
            submit_btn.pack(pady=10)

        # Create a separate frame for the control buttons
        control_frame = tk.Frame(container_window)
        control_frame.pack(fill='x', padx=10, pady=10)

        # Button to add more container rows
        add_another_btn = tk.Button(control_frame, text="Add Another Container", command=add_container_row)
        add_another_btn.pack(side='left', padx=5)

        # Button to add containers from a list of tuples
        add_from_list_btn = tk.Button(control_frame, text="Add Containers from List", command=add_containers_from_list)
        add_from_list_btn.pack(side='left', padx=5)

        # Submit button
        submit_btn = tk.Button(control_frame, text="Submit All Containers",
                               command=lambda: self.submit_containers(container_window, container_entries))
        submit_btn.pack(side='right', padx=5)

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
                    # For export, destination should not be a dummy depot
                    if self.nodes[origin].type not in ['depot', 'depot_arr'] or self.nodes[destination].type != 'terminal':
                        raise ValueError(f"Container {idx}: Export containers must originate from a depot/dummy and be destined to a terminal.")
                elif container_type == 'I':
                    # For import, origin should be a terminal and destination a dummy depot
                    if self.nodes[origin].type != 'terminal' or self.nodes[destination].type != 'depot_arr':
                        raise ValueError(f"Container {idx}: Import containers must originate from a terminal and be destined to a dummy depot.")
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

                # Additional validation: Closing date must be after opening date
                if closing <= opening:
                    raise ValueError(f"Container {idx}: Closing date must be after opening date.")

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

        tk.Label(barge_window, text="Origin Depot ID:").grid(row=2, column=0, padx=5, pady=5, sticky='e')

        # Origin Depot Selection
        depots = self.get_node_ids_by_type('depot')  # List of depots as strings
        if not depots:
            messagebox.showerror("No Depots", "Please add depots before adding a barge.")
            barge_window.destroy()
            return

        origin_var = tk.StringVar(barge_window)
        origin_menu = tk.OptionMenu(barge_window, origin_var, *depots)
        origin_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        origin_var.set(depots[0])  # Set default to first depot

        # Submit button
        submit_btn = tk.Button(barge_window, text="Add", command=lambda: self.submit_barge(barge_window, capacity_entry.get(), fixed_cost_entry.get(), origin_var.get()))
        submit_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def submit_barge(self, window, capacity, fixed_cost, origin):
        try:
            capacity = float(capacity)
            fixed_cost = float(fixed_cost)
            origin = int(origin)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid barge data.")
            return

        # Ensure the origin is a depot
        if self.nodes[origin].type != 'depot':
            messagebox.showerror("Invalid Origin", "Barge origin must be a depot.")
            return

        barge_id = self.next_barge_id
        self.next_barge_id += 1
        barge = Barge(barge_id, capacity, fixed_cost, origin)
        self.barges[barge_id] = barge
        messagebox.showinfo("Barge Added", f"Barge {barge_id} added with Origin Depot {origin}.")
        window.destroy()

    def add_truck(self):
        # Trucks are predefined; no addition allowed
        messagebox.showinfo("Add Truck", "Only two trucks (ID 1 and ID 2) are allowed. Please use 'Edit Trucks' to modify their details.")


    def add_truck_button_disabled(self):
        # Optionally, hide or disable the "Add Truck" button
        pass  # Already handled by not creating the button

    def add_truck(self):
        # Trucks are predefined; no addition allowed
        messagebox.showinfo("Add Truck", "Only two trucks (ID 1 and ID 2) are allowed. Please use 'Edit Trucks' to modify their details.")

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
            barges_text.insert(tk.END, f"ID: {barge.id}, Capacity: {barge.capacity}, Fixed Cost: {barge.fixed_cost}, Origin Depot: {barge.origin}\n")

        # Trucks Tab
        trucks_frame = ttk.Frame(notebook)
        notebook.add(trucks_frame, text="Trucks")
        trucks_text = tk.Text(trucks_frame, width=80, height=10)
        trucks_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trucks_scroll = tk.Scrollbar(trucks_frame, command=trucks_text.yview)
        trucks_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        trucks_text.config(yscrollcommand=trucks_scroll.set)
        for truck in self.trucks.values():
            trucks_text.insert(tk.END, f"Cost per Container: {truck.cost_per_container}\n")

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
                "fixed_cost": barge.fixed_cost,
                "origin": barge.origin  # Save origin depot ID
            } for barge_id, barge in self.barges.items()},
            "trucks": {truck_id: {
                "cost_per_container": truck.cost_per_container
            } for truck_id, truck in self.trucks.items()},
            "depot_to_dummy": self.depot_to_dummy,  # Save depot_to_dummy mapping
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

    def clear_data(self):
        # Clear all data
        self.nodes.clear()
        self.containers.clear()
        self.barges.clear()
        self.trucks.clear()
        self.depot_to_dummy.clear()  # Clear depot_to_dummy mapping
        self.next_node_id = 0
        self.next_container_id = 1
        self.next_barge_id = 1
        self.next_truck_id = 11  # Reset to 3 since 1 and 2 are predefined

        # Re-predefine the two trucks
        self.predefine_trucks()

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

        # Prompt user for maximum optimization time in seconds
        max_time = simpledialog.askinteger(
            "Max Optimization Time",
            "Enter the maximum optimization time in seconds:",
            minvalue=10,  # Minimum 10 sec
            maxvalue=3600  # Maximum 1 hour
        )
        if not max_time:
            return  # User cancelled the dialog

        # Create a progress window
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Solving Model")
        progress_window.geometry("400x100")
        tk.Label(progress_window, text="Optimization in progress...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = max_time
        progress_bar["value"] = 0

        # Event to signal optimization completion
        opt_done = threading.Event()

        # Dictionary to store optimization results
        self.optimization_results = {}

        # Define the optimization thread
        def optimization_thread():
            try:
                nodes_dict, arcs_list = self.construct_arcs()
                model, variables = self.build_and_solve_model(nodes_dict, arcs_list, max_time)
                if model is not None and variables is not None:
                    self.optimization_results['model'] = model
                    self.optimization_results['variables'] = variables
            except Exception as e:
                self.optimization_results['error'] = str(e)
            finally:
                opt_done.set()

        # Start the optimization in a separate thread
        thread = threading.Thread(target=optimization_thread)
        thread.start()

        # Start time tracking
        start_time = time.time()

        # Function to update the progress bar
        def update_progress():
            if opt_done.is_set():
                progress_bar["value"] = max_time
                progress_window.destroy()
                error = self.optimization_results.get('error')
                if error:
                    messagebox.showerror("Optimization Error", f"An error occurred:\n{error}")
                    return
                model = self.optimization_results.get('model')
                variables = self.optimization_results.get('variables')
                if model and variables:
                    if model.Status == GRB.OPTIMAL:
                        messagebox.showinfo("Optimization Completed", "An optimal solution was found.")
                    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
                        messagebox.showwarning("Time Limit Reached",
                                               "Optimization reached the time limit. A feasible solution was found.")
                    # Display results regardless of status
                    self.display_results(model, variables)
                    self.visualize_routes(variables)
                return
            else:
                elapsed_time = time.time() - start_time
                progress_bar["value"] = min(elapsed_time, max_time)
                if elapsed_time < max_time:
                    self.master.after(1000, update_progress)  # Update every second
                else:
                    # Time limit reached; Gurobi should have stopped
                    progress_window.destroy()
                    # Handle cases where optimization might not have set opt_done yet
                    if not opt_done.is_set():
                        messagebox.showwarning("Time Limit Reached", "Optimization reached the maximum time limit.")
                        opt_done.set()
                    return

        # Initiate the progress bar update loop
        update_progress()

    def build_and_solve_model(self, nodes, arcs,max_time):
        try:
            logging.info("Initializing the Gurobi model...")
            model = Model("BargeScheduling")

            # Big M
            M = 2000  # A large constant used in Big M method for conditional constraints
            logging.debug(f"Big M value set to {M}")

            # Define sets
            N = list(nodes.keys())  # Set of all node IDs
            C = list(self.containers.keys())  # Set of all container IDs
            E = [c.id for c in self.containers.values() if c.type == 'E']  # Export containers
            I = [c.id for c in self.containers.values() if c.type == 'I']  # Import containers
            K = list(self.barges.keys()) + [9, 10]  # Set of barges and trucks (IDs 1 and 2)
            KB = list(self.barges.keys())  # Set of barges only

            logging.debug(
                f"Sets defined - Nodes: {N}, Containers: {C}, Export Containers: {E}, Import Containers: {I}, Vehicles: {K}, Barges: {KB}")

            # Define parameters
            Wc = {c.id: c.size for c in self.containers.values()}  # Wc: Container sizes
            Rc = {c.id: c.release_date for c in self.containers.values() if
                  c.type == 'E'}  # Rc: Release dates for export containers
            Oc = {c.id: c.opening_date for c in self.containers.values()}  # Oc: Opening dates for all containers
            Dc = {c.id: c.closing_date for c in self.containers.values()}  # Dc: Closing dates for all containers

            logging.debug("Parameters defined - Wc, Rc, Oc, Dc")

            # Zcj: Indicator if container c is associated with node j
            Zcj = {}
            for c in self.containers.values():
                for j in N:
                    if c.origin == j:
                        Zcj[c.id, j] = 1
                    elif c.destination == j:
                        Zcj[c.id, j] = 1
                    else:
                        Zcj[c.id, j] = 0
            logging.debug("Zcj indicators set for container-node associations")

            HBk = {k: self.barges[k].fixed_cost for k in self.barges.keys()}  # HBk: Fixed costs for each barge
            Qk = {k: self.barges[k].capacity for k in self.barges.keys()}  # Qk: Capacities for each barge
            Or = {k: self.barges[k].origin for k in self.barges.keys()}  # Origin for each barge

            logging.debug("Barge parameters defined - HBk, Qk, Or")

            # depot_to_dummy mapping
            depot_to_dummy = self.depot_to_dummy  # Assuming it's already populated
            logging.debug(f"Depot to Dummy Depot mapping: {depot_to_dummy}")

            # Update arcs_list to include only Arc objects
            arcs_list = arcs  # List of Arc instances
            logging.debug(f"Number of arcs constructed: {len(arcs_list)}")

            Tij = {(arc.origin, arc.destination): arc.travel_time for arc in arcs}  # Tij: Travel times between nodes
            logging.debug("Travel times (Tij) between nodes computed")

            # Handling time and penalty
            L = 20  # Handling time per container in hours (e.g., loading/unloading time)
            gamma = 50  # Penalty cost for visiting sea terminals
            logging.debug(f"Handling time (L): {L} hours, Penalty (gamma): {gamma}")

            # =========================================================================================================================
            #  Define Decision Variables
            # =========================================================================================================================

            logging.info("Defining decision variables...")

            # f_ck: Binary variable indicating if container c is assigned to vehicle k
            f_ck = {}
            for c in C:
                for k in K:
                    f_ck[c, k] = model.addVar(vtype=GRB.BINARY, name=f"f_{c}_{k}")
            logging.debug(f"Defined f_ck variables for {len(f_ck)} container-vehicle pairs")

            # x_ijk: Binary variable indicating if barge k traverses arc (i, j)
            x_ijk = {}
            for k in KB:
                x_ijk[k] = {}
                for arc in arcs_list:
                    i, j, travel_time = arc.origin, arc.destination, arc.travel_time
                    if i != j:
                        x_ijk[k][(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
            logging.debug(f"Defined x_ijk variables for {len(x_ijk)} barges and their possible arcs")

            # p_jk: Continuous variable representing import quantities loaded by barge k at terminal j
            # d_jk: Continuous variable representing export quantities unloaded by barge k at terminal j
            p_jk = {}
            d_jk = {}
            for k in KB:
                for j in N:
                    if self.nodes[j].type == 'terminal':
                        p_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{j}_{k}")
                        d_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_{j}_{k}")
            logging.debug(f"Defined p_jk and d_jk variables for barges at terminal nodes")

            # y_ijk: Continuous variable for import containers on arc (i, j) by barge k
            # z_ijk: Continuous variable for export containers on arc (i, j) by barge k
            y_ijk = {}
            z_ijk = {}
            for k in KB:
                y_ijk[k] = {}
                z_ijk[k] = {}
                for arc in arcs_list:
                    i, j, travel_time = arc.origin, arc.destination, arc.travel_time
                    if i != j:
                        y_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{i}_{j}_{k}")
                        z_ijk[k][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}_{k}")
            logging.debug(f"Defined y_ijk and z_ijk variables for barges on all arcs")

            # t_jk: Continuous variable representing the arrival time of barge k at node j
            t_jk = {}
            for k in KB:
                for j in N:
                    t_jk[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{j}_{k}")
            logging.debug(f"Defined t_jk variables for barges at terminal nodes")

            logging.debug(t_jk)

            # =========================================================================================================================
            #  Define Objective Function
            # =========================================================================================================================

            logging.info("Defining objective function...")

            # Corrected Objective Function
            model.setObjective(
                quicksum(f_ck[c, k] * 2000 for c in C for k in [9, 10]) +  # Truck costs
                quicksum(x_ijk[k][(i, j)] * HBk[k] for k in KB for (i, j) in x_ijk[k]) +  # Barge fixed costs
                quicksum(Tij[(i, j)] * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k]) +  # Barge travel time costs
                quicksum(
                    gamma * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k] if self.nodes[i].type == "terminal"
                ),  # Penalty for visiting sea terminals
                GRB.MINIMIZE
            )
            logging.debug("Objective function defined")

            # =========================================================================================================================
            #  Define Constraints
            # =========================================================================================================================

            logging.info("Defining constraints...")

            # (1) Each container is allocated to exactly one barge or truck
            for c in C:
                model.addConstr(
                    quicksum(f_ck[c, k] for k in K) == 1,
                    name=f"Assignment_{c}"
                )
            logging.debug(f"Defined assignment constraints for {len(C)} containers")

            # (2) Flow conservation for x_ijk (Barge Routes)
            for k in KB:
                origin_node = Or[k]  # Get the origin node for barge k
                try:
                    destination_node = depot_to_dummy[origin_node]  # Map to the corresponding dummy depot node
                    logging.debug(f"Barge {k} originates from Depot {origin_node} to Dummy Depot {destination_node}")
                except KeyError:
                    logging.error(f"No dummy depot found for depot {origin_node}. Check depot_to_dummy mapping.")
                    messagebox.showerror("Mapping Error", f"No dummy depot found for depot {origin_node}.")
                    return None, None

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
            logging.debug("Flow conservation constraints defined for all barges and nodes")

            # (3) Each barge is used at most once
            for k in KB:
                model.addConstr(
                    quicksum(x_ijk[k][(i, j)] for (i, j) in x_ijk[k] if self.nodes[i].type == "depot") <= 1,
                    name=f"Barge_used_{k}"
                )
            logging.debug("Usage constraints defined for barges")

            # (31) Ensure barges only carry containers from their origin depot
            for k in KB:
                origin_node = Or[k]
                for c in C:
                    if self.containers[c].origin != origin_node and self.containers[c].type == 'E':
                        model.addConstr(f_ck[c, k] == 0, name=f"Origin_constraint_{c}_{k}")
            logging.debug("Origin constraints for barges defined")

            # (32) Ensure barges traverse destination nodes if carrying containers
            for c in E + I:
                destination = self.containers[c].destination
                for k in KB:
                    model.addConstr(
                        quicksum(x_ijk[k][(i, destination)] for i in N if (i, destination) in Tij) >= f_ck[c, k],
                        name=f"Barge_{k}_traverse_destination_{c}"
                    )
            logging.debug("Destination traversal constraints for barges defined")

            # (4) Import quantities loaded by barge k at sea terminal j
            for k in KB:
                for j in N:
                    if self.nodes[j].type == "terminal":
                        model.addConstr(
                            p_jk[j, k] == quicksum(Wc[c] * Zcj[c, j] * f_ck[c, k] for c in I),
                            name=f"import_quantities_{j}_{k}"
                        )
            logging.debug("Import quantity constraints defined for barges at terminals")

            # (5) Export quantities loaded by barge k at sea terminal j
            for k in KB:
                for j in N:
                    if self.nodes[j].type == "terminal":
                        model.addConstr(
                            d_jk[j, k] == quicksum(Wc[c] * Zcj[c, j] * f_ck[c, k] for c in E),
                            name=f"Export_quantities_{j}_{k}"
                        )
            logging.debug("Export quantity constraints defined for barges at terminals")

            # (6) Flow equations for y_ijk (import containers)
            for k in KB:
                for j in N:
                    if self.nodes[j].type == 'terminal':
                        inflow = quicksum(y_ijk[k][(i, j)] for i in N if (i, j) in Tij)
                        outflow = quicksum(y_ijk[k][(j, i)] for i in N if (j, i) in Tij)
                        model.addConstr(
                            inflow - outflow == p_jk[j, k],
                            name=f"ImportFlow_{j}_{k}"
                        )
            logging.debug("Import flow constraints defined for barges at terminals")

            # (7) Flow equations for z_ijk (export containers)
            for k in KB:
                for j in N:
                    if self.nodes[j].type == 'terminal':
                        inflow = quicksum(z_ijk[k][(i, j)] for i in N if (i, j) in Tij)
                        outflow = quicksum(z_ijk[k][(j, i)] for i in N if (j, i) in Tij)
                        model.addConstr(
                            inflow - outflow == d_jk[j, k],
                            name=f"ExportFlow_{j}_{k}"
                        )
            logging.debug("Export flow constraints defined for barges at terminals")

            # (8) Capacity constraints for barges and trucks
            for k in K:
                if k in KB:
                    # Barge capacity on each arc
                    for (i, j) in x_ijk[k]:
                        model.addConstr(
                            y_ijk[k][(i, j)] + z_ijk[k][(i, j)] <= Qk[k] * x_ijk[k][(i, j)],
                            name=f"Capacity_{i}_{j}_{k}"
                        )
            logging.debug("Capacity constraints defined for barges and trucks")

            # (9) Barge departure time after release of export containers
            for c in E:
                for k in KB:
                    if c in Rc:
                        depot = self.containers[c].origin
                        model.addConstr(
                            t_jk[depot, k] >= Rc[c] * f_ck[c, k],
                            name=f"BargeDeparture_{c}_{k}"
                        )
            logging.debug("Barge departure time constraints defined for export containers")

            # (10) Time consistency constraints
            for k in KB:
                for arc in arcs_list:
                    i, j, travel_time = arc.origin, arc.destination, arc.travel_time
                    if (i, j) in Tij:
                        model.addConstr(
                            t_jk[j, k] >= t_jk[i, k] + L * quicksum(Zcj[c, i] * f_ck[c, k] for c in C) + Tij[(i, j)] - (
                                    1 - x_ijk[k][(i, j)]) * M,
                            name=f"TimeLB_{i}_{j}_{k}"
                        )
                        model.addConstr(
                            t_jk[j, k] <= t_jk[i, k] + L * quicksum(Zcj[c, i] * f_ck[c, k] for c in C) + Tij[(i, j)] + (
                                    1 - x_ijk[k][(i, j)]) * M,
                            name=f"TimeUB_{i}_{j}_{k}"
                        )
            logging.debug("Time consistency constraints defined for all arcs and barges")

            # (12) Time window constraints for containers at terminals
            for c in C:
                for j in N:
                    if self.nodes[j].type == 'terminal':
                        for k in KB:
                            model.addConstr(
                                t_jk[j, k] >= Oc[c] * Zcj[c, j] - (1 - f_ck[c, k]) * M,
                                name=f"ReleaseTime_{c}_{j}_{k}"
                            )
                            model.addConstr(
                                t_jk[j, k] <= Dc[c] + (1 - f_ck[c, k]) * M,
                                name=f"ClosingTime_{c}_{j}_{k}"
                            )
            logging.debug("Time window constraints defined for containers at terminals")

            # (13) Time of delivery is after time of pickup
            for c in C:
                origin = self.containers[c].origin
                destination = self.containers[c].destination
                for k in KB:
                    model.addConstr(
                        t_jk[destination, k] >= t_jk[origin, k] - (1 - f_ck[c, k]) * M,
                        name=f"Sequence_origin_before_destination_indirect_{k}_{c}"
                    )
            logging.debug("Delivery time constraints defined for containers")

            # =========================================================================================================================
            #  Optimize the Model
            # =========================================================================================================================

            logging.info("Updating and optimizing the model...")
            # Update the model with all variables and constraints
            model.update()

            # Set Gurobi parameters
            model.setParam('OutputFlag', True)  # Enable solver output
            model.setParam('TimeLimit', max_time)  # Set a time limit of 5 minutes (300 seconds)

            # Start the optimization process
            model.optimize()
            logging.info("Optimization process started")

            variables = self.extract_variables(model, nodes, arcs)

            if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                logging.info("Optimization completed with a feasible solution.")

            elif model.Status == GRB.TIME_LIMIT:
                logging.warning("Optimization reached the time limit without finding a feasible solution.")
                messagebox.showwarning("Time Limit Reached",
                                       "Optimization reached the maximum time limit without finding a feasible solution.")
            elif model.Status == GRB.INFEASIBLE:
                messagebox.showerror("Optimization Error", "The model is infeasible; please check your constraints.")
                logging.error("Optimization failed: Model is infeasible.")
                return None, None

            return model, variables

        except Exception as e:
            logging.exception("An unexpected error occurred during model building or optimization.")
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{str(e)}")
            return None, None

    def extract_variables(self, model, nodes, arcs):
        """
        Extracts the necessary variables from the optimized model.
        Args:
            model (gurobipy.Model): The optimized Gurobi model.
            nodes (dict): Dictionary of Node objects.
            arcs (list): List of Arc objects.
        Returns:
            variables (dict): Dictionary containing variable values.
        """
        variables = {}

        # Extract f_ck variables
        f_ck = {}
        for v in model.getVars():
            if v.varName.startswith("f_"):
                parts = v.varName.split("_")
                if len(parts) >= 3:
                    c = int(parts[1])
                    k = parts[2]
                    if k.isdigit():
                        k = int(k)
                    f_ck[c, k] = v.X
        variables['f_ck'] = f_ck

        # Extract x_ijk variables
        x_ijk = {}
        for v in model.getVars():
            if v.varName.startswith("x_"):
                parts = v.varName.split("_")
                if len(parts) >= 4:
                    i = int(parts[1])
                    j = int(parts[2])
                    k = parts[3]
                    if k.isdigit():
                        k = int(k)
                    if k not in x_ijk:
                        x_ijk[k] = {}
                    x_ijk[k][(i, j)] = v.X
        variables['x_ijk'] = x_ijk

        # Extract p_jk variables
        p_jk = {}
        for v in model.getVars():
            if v.varName.startswith("p_"):
                parts = v.varName.split("_")
                if len(parts) >= 3:
                    j = int(parts[1])
                    k = parts[2]
                    if k.isdigit():
                        k = int(k)
                    p_jk[j, k] = v.X
        variables['p_jk'] = p_jk

        # Extract d_jk variables
        d_jk = {}
        for v in model.getVars():
            if v.varName.startswith("d_"):
                parts = v.varName.split("_")
                if len(parts) >= 3:
                    j = int(parts[1])
                    k = parts[2]
                    if k.isdigit():
                        k = int(k)
                    d_jk[j, k] = v.X
        variables['d_jk'] = d_jk

        # Extract y_ijk variables
        y_ijk = {}
        for v in model.getVars():
            if v.varName.startswith("y_"):
                parts = v.varName.split("_")
                if len(parts) >= 4:
                    i = int(parts[1])
                    j = int(parts[2])
                    k = parts[3]
                    if k.isdigit():
                        k = int(k)
                    if k not in y_ijk:
                        y_ijk[k] = {}
                    y_ijk[k][(i, j)] = v.X
        variables['y_ijk'] = y_ijk

        # Extract z_ijk variables
        z_ijk = {}
        for v in model.getVars():
            if v.varName.startswith("z_"):
                parts = v.varName.split("_")
                if len(parts) >= 4:
                    i = int(parts[1])
                    j = int(parts[2])
                    k = parts[3]
                    if k.isdigit():
                        k = int(k)
                    if k not in z_ijk:
                        z_ijk[k] = {}
                    z_ijk[k][(i, j)] = v.X
        variables['z_ijk'] = z_ijk

        # Extract t_jk variables
        t_jk = {}
        for v in model.getVars():
            if v.varName.startswith("t_"):
                parts = v.varName.split("_")
                if len(parts) >= 3:
                    j = int(parts[1])
                    k = parts[2]
                    if k.isdigit():
                        k = int(k)
                    t_jk[j, k] = v.X
        variables['t_jk'] = t_jk

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
                if k in [9, 10]:
                    allocations_text.insert(tk.END, f"Container {c} is allocated to Truck {k}\n")
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
        # Iterate through trucks and find their assignments
        for k in [9, 10]:
            # Find containers allocated to this truck
            allocated_containers = [c for (c, veh), val in variables['f_ck'].items() if veh == k and val > 0.5]
            if allocated_containers:
                truck_routes_text.insert(tk.END, f"\nTruck {k} Routes:\n")
                for c in allocated_containers:
                    origin = self.nodes[self.containers[c].origin]
                    destination = self.nodes[self.containers[c].destination]
                    # Draw the route description
                    truck_routes_text.insert(tk.END, f"Container {c}: {origin.id} -> {destination.id}\n")
            else:
                truck_routes_text.insert(tk.END, f"\nTruck {k} Routes:\nNo containers assigned.\n")

    def get_unique_color(self, index):
        """
        Generates a unique color based on the index.
        If the index exceeds the predefined color list, generates a new color.
        """
        if index < len(self.color_list):
            return self.color_list[index]
        else:
            # Generate a color using HSV and convert it to RGB
            hue = (index * 0.618033988749895) % 1  # Golden ratio conjugate for even distribution
            lightness = 0.5
            saturation = 0.5
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            # Convert to hexadecimal color code
            return '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def visualize_routes(self, variables):
        """
        Visualizes the barge and truck routes on the canvas.
        Args:
            variables (dict): Dictionary containing variable values.
        """
        # Remove existing route lines and truck routes
        self.canvas.delete("route")
        self.canvas.delete("truck_route")

        # Dictionaries to keep track of label counts per node
        start_label_counts = defaultdict(int)
        end_label_counts = defaultdict(int)

        # Define the offset distance between labels
        label_offset = 15  # pixels

        # Assign colors to barges if not already assigned
        for idx, k in enumerate(self.barges.keys()):
            if k not in self.barge_colors:
                self.barge_colors[k] = self.get_unique_color(idx)

        # Visualize Barge Routes
        for k, arcs in variables.get('x_ijk', {}).items():
            barge_color = self.barge_colors.get(k, 'black')  # Default to black if color not found
            for (i, j), val in arcs.items():
                if val > 0.5:
                    origin = self.nodes.get(i)
                    destination = self.nodes.get(j)
                    if origin and destination:
                        # Draw a line for this arc with the barge's color
                        self.canvas.create_line(
                            origin.x, origin.y,
                            destination.x, destination.y,
                            fill=barge_color, width=2,
                            tags="route"
                        )
                    else:
                        logging.warning(f"Origin ({i}) or Destination ({j}) node not found for Barge {k}.")

            # Optionally, mark start and end points for each barge
            origin_node_id = self.barges[k].origin
            dummy_node_id = self.depot_to_dummy.get(origin_node_id)
            if dummy_node_id and dummy_node_id in self.nodes:
                end_node = self.nodes[dummy_node_id]
            else:
                # Fallback: last node in route
                if arcs:
                    last_route = max(arcs.items(), key=lambda item: item[1])[0]
                    end_node = self.nodes.get(last_route[1], self.nodes.get(origin_node_id))
                else:
                    end_node = self.nodes.get(origin_node_id)
            start_node = self.nodes.get(origin_node_id)

            if not start_node or not end_node:
                logging.warning(f"Start or End node not found for Barge {k}.")
                continue

            # Draw start node marker
            self.canvas.create_oval(
                start_node.x - 7, start_node.y - 7,
                start_node.x + 7, start_node.y + 7,
                outline='black', width=2, tags="route"
            )

            # Calculate offset for start label
            start_count = start_label_counts[start_node.id]
            start_label_y = start_node.y - 10 - (start_count * label_offset)
            start_label_text = f"Barge {k} Start"
            self.canvas.create_text(
                start_node.x, start_label_y,
                text=start_label_text,
                fill=barge_color,  # Use barge color for text
                font=('Arial', 10, 'bold'),
                tags="route"
            )
            start_label_counts[start_node.id] += 1

            # Draw end node marker
            self.canvas.create_oval(
                end_node.x - 7, end_node.y - 7,
                end_node.x + 7, end_node.y + 7,
                outline='black', width=2, tags="route"
            )

            # Calculate offset for end label
            end_count = end_label_counts[end_node.id]
            end_label_y = end_node.y - 10 - (end_count * label_offset)
            end_label_text = f"Barge {k} End"
            self.canvas.create_text(
                end_node.x, end_label_y,
                text=end_label_text,
                fill=barge_color,  # Use barge color for text
                font=('Arial', 10, 'bold'),
                tags="route"
            )
            end_label_counts[end_node.id] += 1



        # Notify the user
        messagebox.showinfo("Routes Visualized", "Barge routes have been visualized on the map.")

    def construct_arcs(self):
        """
        Constructs the arcs based on nodes' pixel coordinates.
        Returns:
            nodes (dict): Dictionary of Node objects keyed by node ID.
            arcs (list): List of Arc objects representing possible routes.
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
                    distance = sqrt((origin.x - destination.x) ** 2 + (origin.y - destination.y) ** 2)  # pixels
                    travel_time = distance / average_speed_pixels_per_min  # minutes
                    arc = Arc(origin=i, destination=j, travel_time=travel_time)
                    arcs.append(arc)
        return nodes, arcs


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

        # Load depot_to_dummy mapping
        self.depot_to_dummy = {int(k): int(v) for k, v in data.get("depot_to_dummy", {}).items()}

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
            barge = Barge(barge_id, barge_data["capacity"], barge_data["fixed_cost"], barge_data["origin"])
            self.barges[barge_id] = barge

        # Load trucks
        for truck_id, truck_data in data.get("trucks", {}).items():
            truck_id = int(truck_id)
            truck = Truck(truck_data["cost_per_container"])
            self.trucks[truck_id] = truck

        # Load next IDs
        self.next_node_id = data.get("next_ids", {}).get("node", self.next_node_id)
        self.next_container_id = data.get("next_ids", {}).get("container", self.next_container_id)
        self.next_barge_id = data.get("next_ids", {}).get("barge", self.next_barge_id)
        self.next_truck_id = data.get("next_ids", {}).get("truck", self.next_truck_id)

        messagebox.showinfo("Loaded", f"Network data loaded from {load_path}.")

    def random_generation_containers(self, container_amount=10, buffer_time=24*60):
        """
        Generates random containers based on current depots and terminals.
        Args:
            container_amount (int): Number of containers to generate.
            buffer_time (int): Minimum buffer time in minutes between opening and closing dates.
        """
        if not self.nodes:
            messagebox.showerror("No Nodes", "Please add nodes before generating containers.")
            return

        # Identify depots, dummy depots, and terminals
        depots = [node_id for node_id, node in self.nodes.items() if node.type == 'depot']
        dummy_depots = [node_id for node_id, node in self.nodes.items() if node.type == 'depot_arr']
        terminals = [node_id for node_id, node in self.nodes.items() if node.type == 'terminal']

        if not depots or not dummy_depots or not terminals:
            messagebox.showerror("Insufficient Nodes", "Ensure that there are depots, dummy depots, and terminals in the network.")
            return

        # Ensure the number of dummy depots matches the number of depots
        if len(dummies := [dummy for dummy in dummy_depots]) != len(depots):
            messagebox.showerror("Mismatch", "Each depot must have a corresponding dummy depot.")
            return

        # Seed for reproducibility (optional)
        random.seed(42)

        for _ in range(container_amount):
            container_id = self.next_container_id
            self.next_container_id += 1

            size = random.choice([1, 2])  # Size: 1 or 2
            container_type = random.choice(["E", "I"])

            if container_type == "E":
                # Opening date within first 24 hours
                opening_date = random.randint(0, 24*60)  # 0 to 1440 minutes
                # Closing date is at least buffer_time after opening_date
                max_closing_date = opening_date + buffer_time + 172*60  # Up to ~3 days
                closing_date = opening_date + buffer_time + random.randint(0, max_closing_date - (opening_date + buffer_time))
                # Release date is before or at opening_date
                release_date = random.randint(0, opening_date)

                origin = random.choice(depots)  # Random depot origin
                destination = random.choice(terminals)  # Random terminal destination

            else:  # Import containers
                release_date = None
                # Opening date within first 24 hours
                opening_date = random.randint(0, 24*60)  # 0 to 1440 minutes
                # Closing date is at least buffer_time after opening_date
                max_closing_date = opening_date + buffer_time + 172*60  # Up to ~3 days
                closing_date = opening_date + buffer_time + random.randint(0, max_closing_date - (opening_date + buffer_time))

                origin = random.choice(terminals)  # Random terminal origin
                destination = random.choice(dummy_depots)  # Random dummy depot destination

            # Cap closing_date to maximum allowed (e.g., 196*60 minutes)
            closing_date = min(closing_date, 196*60)

            # Create and add the container
            container = Container(container_id, size, release_date, opening_date, closing_date, origin, destination, container_type)
            self.containers[container_id] = container

        messagebox.showinfo("Random Containers Added", f"{container_amount} random containers have been generated successfully.")

    def generate_random_containers(self):
        # Prompt user for the number of containers to generate
        container_amount = simpledialog.askinteger("Random Containers", "Enter the number of random containers to generate:", minvalue=1, maxvalue=1000)
        if container_amount:
            self.random_generation_containers(container_amount=container_amount)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkCreatorApp(root)
    root.mainloop()
