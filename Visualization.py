
def visualize_routes(nodes, barges, variables, containers, node_coords):
    import folium
    from geopy.distance import geodesic
    from pyproj import Geod
    import folium
    from folium.plugins import BeautifyIcon

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

    # Function to offset coordinates slightly to avoid overlapping
    def offset_coords(coords, offset_meters):
        geod = Geod(ellps='WGS84')
        lat, lon = coords  # Coordinates are in (lat, lon)
        # Offset by moving north and east by a small amount
        azimuth = 45  # Northeast direction
        lon_offset, lat_offset, _ = geod.fwd(lon, lat, azimuth, offset_meters)
        return [lat_offset, lon_offset]  # Return in (lat, lon) order

    # Plot barge routes with labels
    for k in barges.keys():
        visits = [(i, j) for (i, j), var in x_ijk[k].items() if var.X > 0.5]
        if not visits:
            continue  # Skip if no route is defined

        # Extract time variables to sort the visits
        t_jk = variables['t_jk']
        # Create a list of visits with their start times
        sorted_visits = sorted(visits, key=lambda arc: t_jk.get((arc[0], k), 0))

        # Reconstruct the route based on sorted times
        route = sorted_visits

        # Extract coordinates in order
        route_coords = []
        for arc in route:
            i, j = arc
            route_coords.append(node_coords[i])
        # Add the last destination
        last_j = route[-1][1]
        route_coords.append(node_coords[last_j])

        # Offset the route slightly to avoid overlapping
        offset_route = [offset_coords(coords, idx * 10) for idx, coords in enumerate(route_coords)]

        folium.PolyLine(
            locations=offset_route,
            color=barge_color_map[k],
            weight=5,
            opacity=0.8,
            tooltip=f"Barge {k}"
        ).add_to(m)
        # Add labels along the route
        for idx in range(len(offset_route) - 1):
            mid_lat = (offset_route[idx][0] + offset_route[idx + 1][0]) / 2
            mid_lon = (offset_route[idx][1] + offset_route[idx + 1][1]) / 2
            folium.map.Marker(
                [mid_lat, mid_lon],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 10pt; color : {barge_color_map[k]};">Barge {k}</div>',
                )
            ).add_to(m)

    # Plot truck routes with labels
    truck_routes = []
    for c in containers.values():
        if f_ck[c.id, 'T'].X > 0.5:
            origin = c.origin
            destination = c.destination
            coords_o = node_coords[origin]
            coords_d = node_coords[destination]
            truck_routes.append((coords_o, coords_d))

    # Offset and plot truck routes
    for idx, (coords_o, coords_d) in enumerate(truck_routes):
        # Offset the route slightly to avoid overlapping
        offset_o = offset_coords(coords_o, idx * 10 + 20)
        offset_d = offset_coords(coords_d, idx * 10 + 20)
        folium.PolyLine(
            locations=[offset_o, offset_d],
            color='black',
            weight=2,
            opacity=0.8,
            dash_array='5, 10',
            tooltip=f"Truck Route"
        ).add_to(m)
        # Add label at midpoint
        mid_lat = (offset_o[0] + offset_d[0]) / 2
        mid_lon = (offset_o[1] + offset_d[1]) / 2
        folium.map.Marker(
            [mid_lat, mid_lon],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 10pt; color : black;">Truck</div>',
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

    # Prepare barge scheduling data
    barge_data = []
    for k in barges.keys():
        # Extract node visits based on x_ijk values
        visits = [(i, j) for (i, j), var in x_ijk[k].items() if var.X > 0.5]
        if not visits:
            continue  # Skip if no route is defined

        # Reconstruct the route in order by following the path
        # Create a mapping from origin to destination
        origin_to_dest = {i: j for (i, j) in visits}

        # Find the starting node (node with outgoing arcs but no incoming arcs for this barge)
        all_origins = set([i for (i, _) in visits])
        all_destinations = set([j for (_, j) in visits])
        start_nodes = all_origins - all_destinations
        if not start_nodes:
            print(f"Barge {k} has no clear starting node.")
            continue
        start_node = start_nodes.pop()

        # Reconstruct the ordered list of arcs
        route = []
        current = start_node
        while current in origin_to_dest:
            next_node = origin_to_dest[current]
            route.append((current, next_node))
            current = next_node

        print(f"Barge {k} Route: {route}")  # Debugging statement

        # Initialize onboard containers
        onboard_containers = set()

        # Add scheduling details to barge data
        for (i, j) in route:
            # **Unload** containers at node i whose destination is i
            for c in containers.values():
                if f_ck[c.id, k].X > 0.5 and c.destination == i:
                    if c.id in onboard_containers:
                        onboard_containers.discard(c.id)

            # **Load** containers at node i whose origin is i
            for c in containers.values():
                if f_ck[c.id, k].X > 0.5 and c.origin == i:
                    onboard_containers.add(c.id)

            # **Unload** containers at node j whose destination is j
            for c in containers.values():
                if f_ck[c.id, k].X > 0.5 and c.destination == j:
                    if c.id in onboard_containers:
                        onboard_containers.discard(c.id)

            # **Load** containers at node j whose origin is j
            for c in containers.values():
                if f_ck[c.id, k].X > 0.5 and c.origin == j:
                    onboard_containers.add(c.id)

            # Build the task description
            start_time = t_jk.get((i, k), 0)
            end_time = t_jk.get((j, k), 0)
            barge_data.append({
                'Resource': f'Barge {k}',
                'Start': start_time,
                'End': end_time,
                'Task': f'{i}→{j} (Containers: {", ".join(map(str, sorted(onboard_containers)))} )'
            })

    # Prepare truck scheduling data
    truck_data = []
    for c in containers.values():
        if f_ck[c.id, 'T'].X > 0.5:  # If truck is used
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
        # Add task labels to the left of the y-axis
        ax.text(
            x=row['Start'] - 5,  # Position well to the left of the bars
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
    ax.set_xlim(left=-10)  # Add space on the left for task labels

    # Enhance layout
    plt.tight_layout()
    plt.show()

