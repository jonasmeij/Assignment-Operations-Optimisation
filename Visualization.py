def visualize_routes(nodes, barges, variables, containers, node_coords):
    import folium

    x_ijk = variables['x_ijk']
    f_ck = variables['f_ck']

    # Create a map centered around Rotterdam
    rotterdam_coords = [51.9244, 4.4777]  # Coordinates of Rotterdam
    m = folium.Map(location=rotterdam_coords, zoom_start=11)

    # Add nodes to the map with different colors for depots and terminals
    for node_id, coords in node_coords.items():
        node = nodes[node_id]
        if node.type == 'depot':
            folium.Marker(
                location=coords,
                popup=f"Depot (Node {node_id})",
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
        elif node.type == 'terminal':
            folium.Marker(
                location=coords,
                popup=f"Terminal (Node {node_id})",
                icon=folium.Icon(color='blue', icon='anchor')
            ).add_to(m)
        elif node.type == 'depot_arr':
            folium.Marker(
                location=coords,
                popup=f"Depot Arrival (Node {node_id})",
                icon=folium.Icon(color='gray')
            ).add_to(m)
        else:
            folium.Marker(
                location=coords,
                popup=f"Node {node_id}",
                icon=folium.Icon(color='gray')
            ).add_to(m)

    # Define colors for barges
    barge_colors = ['purple', 'orange', 'darkred', 'cadetblue', 'green']
    barge_color_map = {}
    for idx, k in enumerate(barges.keys()):
        barge_color_map[k] = barge_colors[idx % len(barge_colors)]

    # Plot barge routes with labels
    for k in barges.keys():
        visits = [(i, j) for (i, j), var in x_ijk[k].items() if var.X > 0.5]
        if not visits:
            continue  # Skip if no route is defined

        # Reconstruct the route based on traversed arcs
        route = visits

        # Extract coordinates in order
        route_coords = []
        for arc in route:
            i, j = arc
            route_coords.append(node_coords[i])
        # Add the last destination
        last_j = route[-1][1]
        route_coords.append(node_coords[last_j])

        # Draw the polyline without offsetting
        folium.PolyLine(
            locations=route_coords,
            color=barge_color_map[k],
            weight=5,
            opacity=0.8,
            tooltip=f"Barge {k}"
        ).add_to(m)
        # Add labels along the route
        for idx in range(len(route_coords) - 1):
            mid_lat = (route_coords[idx][0] + route_coords[idx + 1][0]) / 2
            mid_lon = (route_coords[idx][1] + route_coords[idx + 1][1]) / 2
            folium.Marker(
                [mid_lat, mid_lon],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 10pt; color : {barge_color_map[k]};">Barge {k}</div>',
                )
            ).add_to(m)

    # Plot truck routes with labels indicating number of trucks per route
    # Collect all containers assigned to trucks
    truck_routes = {}
    for c in containers.values():
        if f_ck[c.id, 'T'].X > 0.5:
            origin = c.origin
            destination = c.destination
            route_key = (origin, destination)
            if route_key in truck_routes:
                truck_routes[route_key] += 1
            else:
                truck_routes[route_key] = 1

    # Plot each unique truck route with the count
    for (origin, destination), count in truck_routes.items():
        coords_o = node_coords[origin]
        coords_d = node_coords[destination]
        folium.PolyLine(
            locations=[coords_o, coords_d],
            color='black',
            weight=2,
            opacity=0.2,
            dash_array='5, 10',
            tooltip=f"Truck Route: {count} truck(s)"
        ).add_to(m)
        # Add label at midpoint
        mid_lat = (coords_o[0] + coords_d[0]) / 2
        mid_lon = (coords_o[1] + coords_d[1]) / 2
        folium.Marker(
            [mid_lat, mid_lon],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 10pt; color : black;">{count} Truck(s)</div>',
            )
        ).add_to(m)

    # Add legend
    from branca.element import Template, MacroElement

    legend_html = '''
     {% macro html(this, kwargs) %}

     <div style="
         position: fixed;
         bottom: 50px;
         left: 50px;
         width: 150px;
         height: 200px;
         z-index:9999;
         font-size:14px;
         ">
         <b>Legend</b>
         <ul style="list-style: none; padding: 0;">
             <li><span style="color: black;">&#8212;&#8212;&#8212;&#8212;</span> Truck Route</li>
    '''

    for k in barges.keys():
        color = barge_color_map[k]
        legend_html += f'''
             <li><span style="color: {color};">&#8212;&#8212;&#8212;&#8212;</span> Barge {k}</li>
        '''

    legend_html += '''
         </ul>
     </div>
     {% endmacro %}
    '''

    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)

    # Save the map to an HTML file
    m.save('rotterdam_routes.html')
    print("Map has been saved to 'rotterdam_routes.html'. Open this file in a web browser to view the map.")


def visualize_schedule(nodes, barges, variables, containers):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract variables
    f_ck = variables['f_ck']
    t_jk = variables['t_jk']  # Pre-extracted values
    x_ijk = variables['x_ijk']

    # Identify depot nodes
    depot_nodes = {node_id for node_id, node in nodes.items() if node.type == "depot"}

    if not depot_nodes:
        print("No depot nodes found.")
        return

    # Prepare barge scheduling data
    barge_data = []
    for k in barges.keys():
        # Extract arcs (i,j) for which x_ijk > 0.5
        visits = [(i, j) for (i, j), var in x_ijk[k].items() if var.X > 0.5]
        if not visits:
            continue  # Skip if no route is defined for this barge

        # Create a mapping from origin to destination
        origin_to_dest = {i: j for (i, j) in visits}

        # Since it's a loop and must start/end at a depot, find a depot node that is in the route's origins
        start_node_candidates = depot_nodes.intersection(origin_to_dest.keys())
        if not start_node_candidates:
            print(f"Barge {k}: No depot node found as a starting point in the route.")
            continue

        # Pick one depot as the starting point
        start_node = next(iter(start_node_candidates))

        # Reconstruct the loop route
        route = []
        current = start_node
        visited_count = 0
        while current in origin_to_dest:
            next_node = origin_to_dest[current]
            route.append((current, next_node))
            current = next_node
            visited_count += 1

            # Break if we've come back to the start node, completing the loop
            if current == start_node:
                break

            # Just a safety check to prevent infinite loops if there's no proper cycle
            if visited_count > len(visits):
                print(f"Barge {k}: Could not reconstruct a proper loop route. Possibly malformed data.")
                route = []
                break

        if not route:
            print(f"Barge {k}: Route could not be constructed.")
            continue

        print(f"Barge {k} Route: {route}")  # Debug statement

        # Track containers on board as we move along the route
        onboard_containers = set()

        # Since it's a cycle, we should ensure the times and loading/unloading steps make sense
        for (i, j) in route:
            # Unload containers at node j (if any have j as destination)
            for c in containers.values():
                if f_ck[c.id, k].X > 0.5 and c.destination == j:
                    if c.id in onboard_containers:
                        onboard_containers.discard(c.id)

            # Load containers at node i (if any have i as origin)
            for c in containers.values():
                if f_ck[c.id, k].X > 0.5 and c.origin == i:
                    onboard_containers.add(c.id)

            # Start and end time from pre-calculated t_jk
            start_time = t_jk.get((i, k), 0)
            end_time = t_jk.get((j, k), 0)

            barge_data.append({
                'Resource': f'Barge {k}',
                'Start': start_time,
                'End': end_time,
                'Task': f'{i}→{j} (Containers: {", ".join(map(str, sorted(onboard_containers)))})'
            })

    # Prepare truck scheduling data
    truck_data = []
    for c in containers.values():
        if f_ck[c.id, 'T'].X > 0.5:  # If truck is used
            # Use container release/opening_date as start and closing_date as end if available
            start_time = c.release_date if c.release_date is not None else c.opening_date
            end_time = c.closing_date
            truck_data.append({
                'Resource': 'Truck',
                'Start': start_time,
                'End': end_time,
                'Task': f'Container {c.id} ({c.origin}→{c.destination})'
            })

    # Combine scheduling data into a DataFrame
    schedule_data = barge_data + truck_data
    df = pd.DataFrame(schedule_data)

    if df.empty:
        print("No scheduling data to visualize.")
        return

    # Assign row indices for each task
    df['Row'] = range(len(df))

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Assign colors to resources
    resources = df['Resource'].unique()
    colors = plt.cm.get_cmap('tab20', len(resources))
    color_map = {resource: colors(i) for i, resource in enumerate(resources)}

    # Plot each task as a horizontal bar
    for _, row in df.iterrows():
        ax.barh(
            y=row['Row'],
            width=row['End'] - row['Start'],
            left=row['Start'],
            height=0.6,
            color=color_map[row['Resource']],
            edgecolor='black'
        )
        # Add task labels to the left of the bars
        ax.text(
            x=row['Start'] - 5,
            y=row['Row'],
            s=row['Task'],
            va='center',
            ha='right',
            color='black',
            fontsize=9
        )

    # Customize axis labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_yticks(df['Row'])
    ax.set_yticklabels(df['Resource'])
    ax.set_title('Schedule of Barges and Trucks', fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Adjust limits to make space for labels
    ax.set_xlim(left=-10)

    # Enhance layout
    plt.tight_layout()
    plt.show()

def visualize_schedule_random(nodes, barges, variables, containers):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract variables
    f_ck = variables['f_ck']
    t_jk = variables['t_jk']
    x_ijk = variables['x_ijk']

    # Identify depot nodes
    depot_nodes = {node_id for node_id, node in nodes.items() if node.type == "depot"}

    if not depot_nodes:
        print("No depot nodes found.")
        return

    # Prepare barge scheduling data
    barge_data = []
    for k in barges.keys():
        visits = [(i, j) for (i, j), var in x_ijk[k].items() if var.X > 0.5]
        if not visits:
            continue

        origin_to_dest = {i: j for (i, j) in visits}
        start_node_candidates = depot_nodes.intersection(origin_to_dest.keys())
        if not start_node_candidates:
            continue

        start_node = next(iter(start_node_candidates))
        route = []
        current = start_node
        visited_count = 0
        while current in origin_to_dest:
            next_node = origin_to_dest[current]
            route.append((current, next_node))
            current = next_node
            visited_count += 1
            if current == start_node:
                break
            if visited_count > len(visits):
                route = []
                break

        for (i, j) in route:
            start_time = t_jk.get((i, k), 0)
            end_time = t_jk.get((j, k), 0)
            barge_data.append({
                'Resource': f'Barge {k}',
                'Start': start_time,
                'End': end_time,
                'Task': f'{i}→{j}'
            })

    # Count total containers assigned to trucks
    truck_containers_count = sum(1 for c in containers.values() if f_ck[c.id, 'T'].X > 0.5)

    # Combine barge data into a DataFrame
    df = pd.DataFrame(barge_data)

    if df.empty:
        print("No barge scheduling data to visualize.")
        return

    # Add a row index for plotting
    df['Row'] = range(len(df))

    # Create the plot
    fig, ax = plt.subplots(figsize=(8.27, 5.83))  # A5 landscape size in inches

    # Assign colors to resources
    resources = df['Resource'].unique()
    colors = plt.cm.get_cmap('tab20', len(resources))
    color_map = {resource: colors(i) for i, resource in enumerate(resources)}

    # Plot each task as a horizontal bar
    for _, row in df.iterrows():
        ax.barh(
            y=row['Row'],
            width=row['End'] - row['Start'],
            left=row['Start'],
            height=0.6,
            color=color_map[row['Resource']],
            edgecolor='black'
        )
        # Add a simple task label in the center of each bar
        ax.text(
            x=row['Start'] + (row['End'] - row['Start']) / 2,
            y=row['Row'],
            s=row['Task'],
            va='center',
            ha='center',
            color='white',
            fontsize=8
        )

    # Customize the axes
    ax.set_xlabel('Time', fontsize=10)
    ax.set_yticks(df['Row'])
    ax.set_yticklabels(df['Resource'])
    ax.set_title('Schedule of Barges (A5 Size)', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Add legend for the number of containers assigned to trucks
    legend_text = f'Total Containers Assigned to Trucks: {truck_containers_count}'
    ax.text(
        1.05, 0.5, legend_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
        verticalalignment='center'
    )

    # Final layout adjustments
    plt.tight_layout()
    plt.show()






