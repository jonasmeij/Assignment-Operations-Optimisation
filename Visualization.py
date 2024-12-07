from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import folium
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np
from pyproj import Geod
import random



def visualize_routes(nodes, barges, variables, containers, node_coords,file_name):

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
    m.save(file_name)
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

def visualize_schedule_random(nodes, barges, variables, containers,output_file):
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
                'Task': f'{i}→{j}',
                'Barge': k  # Include barge identifier for grouping later
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

    # Compute the midpoint for each barge schedule
    barge_midpoints = (
        df.groupby('Resource')
        .agg(Midpoint=('Row', 'mean'))
        .reset_index()
    )

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
            color='black',
            fontsize=8
        )

    # Add text for each barge in the middle of its schedule
    for _, midpoint in barge_midpoints.iterrows():
        ax.text(
            x=-50,  # Place the text outside the left side of the plot
            y=midpoint['Midpoint'],
            s=midpoint['Resource'],
            va='center',
            ha='right',
            fontsize=10,
            fontweight='bold'
        )

    # Customize the axes
    ax.set_xlabel('Time', fontsize=10)
    ax.set_yticks(df['Row'])
    ax.set_yticklabels([])  # Remove repeated barge labels from individual rows
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
    plt.savefig(output_file)



def visualize_routes_static(
        nodes, barges, variables, containers, node_coords,
        output_filename_full
):
    """
    Visualize detailed routes on static maps using Cartopy and Matplotlib.
    Generates two maps:
    1. Full Rotterdam map with depots, terminals, cities, landmarks, and barge routes with direction indicators.
    2. Zoomed-in map focusing exclusively on terminals with direction indicators.

    Parameters:
    - nodes: Dictionary of node objects with 'type' attribute.
    - barges: Dictionary of barges.
    - variables: Dictionary containing optimization variables like 'x_ijk' and 'f_ck'.
    - containers: Dictionary of container objects with 'id', 'origin', and 'destination'.
    - node_coords: Dictionary mapping node IDs to (latitude, longitude) tuples.
    - output_filename_full: Filename for the saved full map image.
    - output_filename_terminals: Filename for the saved terminal-focused map image.
    """
    # Extract variables
    x_ijk = variables['x_ijk']
    f_ck = variables['f_ck']

    # Define colors for barges
    barge_colors = [
        'purple', 'orange', 'darkred', 'cadetblue', 'green',
        'magenta', 'cyan', 'brown', 'olive', 'teal'
    ]
    barge_color_map = {}
    for idx, k in enumerate(barges.keys()):
        barge_color_map[k] = barge_colors[idx % len(barge_colors)]

    # Calculate total number of containers transported by truck
    total_trucks = sum(
        1 for c in containers.values()
        if (c.id, 'T') in f_ck and f_ck[(c.id, 'T')].X > 0.5
    )

    # ============================
    # 1. Generate Full Rotterdam Map
    # ============================

    # Create the full map figure
    fig_full = plt.figure(figsize=(15, 15))
    ax_full = plt.axes(projection=ccrs.Mercator())

    # Set the extent around Rotterdam with some padding
    lats_full = [coord[0] for coord in node_coords.values()]
    lons_full = [coord[1] for coord in node_coords.values()]
    buffer_full = 0.1  # degrees
    ax_full.set_extent(
        [min(lons_full) - buffer_full, max(lons_full) + buffer_full,
         min(lats_full) - buffer_full, max(lats_full) + buffer_full],
        crs=ccrs.PlateCarree()
    )

    # Add detailed map features using high-resolution Natural Earth data
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                         edgecolor='face',
                                         facecolor=cfeature.COLORS['water'])
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                             edgecolor='black',
                                             facecolor='none')
    borders = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m',
                                           edgecolor='black',
                                           facecolor='none')
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                                          edgecolor='blue',
                                          facecolor='none')
    roads = cfeature.NaturalEarthFeature('cultural', 'roads', '10m',
                                         edgecolor='gray',
                                         facecolor='none',
                                         linewidth=0.5,
                                         alpha=0.5)

    ax_full.add_feature(land)
    ax_full.add_feature(ocean)
    ax_full.add_feature(coastline)
    ax_full.add_feature(borders)
    ax_full.add_feature(rivers)
    ax_full.add_feature(roads)

    # Define depot to city mapping
    depot_cities = {
        0: 'Veghel',
        1: 'Tilburg',
        2: 'Eindhoven',
        3: 'Nijmegen',
        4: 'Utrecht'
    }

    # Define major cities with their coordinates (latitude, longitude)
    major_cities = {
        'Rotterdam': (51.9244, 4.4777),
        'Amsterdam': (52.3676, 4.9041),
        'The Hague': (52.0705, 4.3007),
        'Den Bosch': (51.6978, 5.3037),
        'Groningen': (53.2194, 6.5665),
        'Maastricht': (50.8514, 5.6900),
        'Arnhem': (51.9851, 5.8987),
        'Leeuwarden': (53.2010, 5.7997),
        'Breda': (51.5719, 4.7683),
        'Apeldoorn': (52.2112, 5.9699),
        # Add more cities as needed
    }

    # Helper function to check if a point is within the map extent
    def is_within_extent(lat, lon, extent):
        west, east, south, north = extent
        return west <= lon <= east and south <= lat <= north

    # Get current map extent
    map_extent = ax_full.get_extent(crs=ccrs.PlateCarree())

    # Plot depots with reduced marker size and city annotations
    for node_id, coords in node_coords.items():
        node = nodes[node_id]
        lat, lon = coords

        # Skip plotting 'depot_arr' nodes
        if node.type == 'depot_arr':
            continue

        if node.type == 'depot':
            city_name = depot_cities.get(node_id, 'Unknown')
            ax_full.plot(
                lon, lat,
                marker='s', markersize=30,  # Depot size reduced to 30
                markeredgecolor='black',
                markerfacecolor='red',
                transform=ccrs.PlateCarree(),
                label='Depot' if 'Depot' not in ax_full.get_legend_handles_labels()[1] else ""
            )
            # Annotate depot with city name
            ax_full.text(
                lon + 0.005, lat + 0.005,
                f'Depot: {city_name}',
                transform=ccrs.PlateCarree(),
                fontsize=9, weight='bold'
            )
        elif node.type == 'terminal':
            ax_full.plot(
                lon, lat,
                marker='^', markersize=20,  # Terminal size set to 20
                markeredgecolor='black',
                markerfacecolor='blue',
                transform=ccrs.PlateCarree(),
                label='Terminal' if 'Terminal' not in ax_full.get_legend_handles_labels()[1] else ""
            )
            # Removed terminal labels on the map
        else:
            ax_full.plot(
                lon, lat,
                marker='o', markersize=30,  # Reduced size
                markeredgecolor='black',
                markerfacecolor='gray',
                transform=ccrs.PlateCarree(),
                label='Node' if 'Node' not in ax_full.get_legend_handles_labels()[1] else ""
            )
            ax_full.text(
                lon + 0.005, lat + 0.005,
                f'Node {node_id}',
                transform=ccrs.PlateCarree(),
                fontsize=9
            )

    # Plot city markers, excluding cities where depots are located and not within map bounds
    for city, (lat, lon) in major_cities.items():
        # Check if the city coincides with any depot
        depot_overlap = False
        for depot_id, depot_city in depot_cities.items():
            depot_coords = node_coords.get(depot_id)
            if depot_coords and (lat, lon) == depot_coords:
                depot_overlap = True
                break
        if depot_overlap:
            continue  # Skip plotting city marker if overlapping with a depot

        # Check if city is within current map extent
        if not is_within_extent(lat, lon, map_extent):
            continue  # Skip plotting if city is outside the map bounds

        # Plot city marker
        ax_full.plot(
            lon, lat,
            marker='*', markersize=15,  # Star marker for cities
            markeredgecolor='black',
            markerfacecolor='gold',
            transform=ccrs.PlateCarree(),
            label='City' if 'City' not in ax_full.get_legend_handles_labels()[1] else ""
        )
        # Annotate city name
        ax_full.text(
            lon + 0.005, lat + 0.005,
            city,
            transform=ccrs.PlateCarree(),
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # Plot barge routes with direction arrows
    for k in barges.keys():
        # Extract the arcs with x_ijk[k][i,j] > 0.5
        if k not in x_ijk:
            continue  # Skip if barge k has no routes defined

        visits = [
            (i, j) for (i, j), var in x_ijk[k].items()
            if var.X > 0.5  # Use var.X for Gurobi
        ]

        if not visits:
            continue  # Skip if no route is defined

        # Reconstruct the route
        route = visits

        # Extract coordinates in order
        route_coords = []
        for arc in route:
            i, j = arc
            route_coords.append(node_coords[i])
        # Add the last destination
        last_j = route[-1][1]
        route_coords.append(node_coords[last_j])

        # Separate latitudes and longitudes
        lats_route = [coord[0] for coord in route_coords]
        lons_route = [coord[1] for coord in route_coords]

        # Plot the route
        ax_full.plot(
            lons_route, lats_route,
            color=barge_color_map[k],
            linewidth=2.5,
            linestyle='-',
            marker=None,
            alpha = 0.7 ,
            transform=ccrs.PlateCarree(),
            label=f'Barge {k}' if f'Barge {k}' not in ax_full.get_legend_handles_labels()[1] else ""
        )

        # Add an arrow to indicate direction
        if len(lons_route) >= 2:
            # Define the number of arrows you want to display along the route
            num_arrows = max(len(lons_route) // 10, 1)  # Adjust the divisor for spacing
            for idx in range(0, len(lons_route) - 1, num_arrows):
                start_lon, start_lat = lons_route[idx], lats_route[idx]
                end_lon, end_lat = lons_route[idx + 1], lats_route[idx + 1]

                # Calculate arrow properties
                dx = end_lon - start_lon
                dy = end_lat - start_lat

                ax_full.annotate(
                    '',  # No text
                    xy=(end_lon, end_lat),  # Arrow end
                    xytext=(start_lon, start_lat),  # Arrow start
                    arrowprops=dict(
                        arrowstyle="->",
                        color=barge_color_map[k],
                        linewidth=1.5,
                        shrinkA=0,
                        shrinkB=0,
                        alpha=0.8
                    ),
                    transform=ccrs.PlateCarree()
                )



    # Create custom legend for full map
    legend_elements_full = []

    # Node types
    legend_elements_full.append(Line2D(
        [0], [0], marker='s', color='w', label='Depot',
        markerfacecolor='red', markersize=10, markeredgecolor='black'
    ))
    legend_elements_full.append(Line2D(
        [0], [0], marker='^', color='w', label='Terminal',
        markerfacecolor='blue', markersize=5, markeredgecolor='black'
    ))
    legend_elements_full.append(Line2D(
        [0], [0], marker='o', color='w', label='Node',
        markerfacecolor='gray', markersize=5, markeredgecolor='black'
    ))
    legend_elements_full.append(Line2D(
        [0], [0], marker='*', color='w', label='City',
        markerfacecolor='gold', markersize=15, markeredgecolor='black'
    ))

    # Barge routes
    for k, color in barge_color_map.items():
        legend_elements_full.append(Line2D(
            [0], [0], color=color, lw=2, label=f'Barge {k}'
        ))

    ax_full.legend(
        handles=legend_elements_full,
        loc='lower left',
        fontsize='small',
        framealpha=0.9
    )

    # Add title with total trucks information
    ax_full.set_title(
        f'Rotterdam Routes Visualization\nTotal Containers Transported by Truck: {total_trucks}',
        fontsize=16,
        weight='bold'
    )

    # Add gridlines
    gl_full = ax_full.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl_full.top_labels = False
    gl_full.right_labels = False

    # Add scale bar
    fontprops = fm.FontProperties(size=10)
    scalebar_full = AnchoredSizeBar(ax_full.transData,
                                    0.01, '1 km', 'lower right',
                                    pad=0.1,
                                    color='black',
                                    frameon=False,
                                    size_vertical=0.005,
                                    fontproperties=fontprops)
    ax_full.add_artist(scalebar_full)

    # Add north arrow
    ax_full.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.90),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center',
                     fontsize=12,
                     xycoords=ax_full.transAxes)

    # Save the full map
    plt.savefig(output_filename_full, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_full)
    print(f"Full map has been saved to '{output_filename_full}'.")


def visualize_routes_terminals(
    nodes,
    barges,
    variables,
    containers,
    node_coords,
    output_file="barge_routes.png",
    seed=None  # Optional seed for reproducibility
):
    """
    Visualizes barge routes on a dynamic geographical map focusing only on terminals.
    Adds starting and ending points for each barge in the top-right corner of the map.
    Assigns a unique color to each barge and includes a comprehensive legend.
    Excludes barges with only one arc. Saves the map as a high-resolution image.

    Args:
        nodes (dict): Dictionary of Node objects.
        barges (dict): Dictionary of Barge objects.
        variables (dict): Dictionary containing optimized variable values.
        containers (dict): Dictionary of Container objects.
        node_coords (dict): Dictionary of node coordinates (latitude, longitude).
        output_file (str): Filename for the output map image.
        seed (int, optional): Seed for random number generator for reproducibility.
    """
    # Initialize Geod with WGS84 ellipsoid for geodesic calculations
    geod = Geod(ellps='WGS84')

    # Set random seed if provided for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Collect terminal nodes and their coordinates
    terminal_nodes = [n for n in nodes if nodes[n].type == "terminal"]
    terminal_coords = [node_coords[n] for n in terminal_nodes]

    if not terminal_coords:
        raise ValueError("No terminal coordinates found.")

    # Determine map bounds based on terminal coordinates with a margin
    lats, lons = zip(*terminal_coords)
    margin = 0.05  # Degrees margin; adjust as needed based on geographical spread
    lat_min, lat_max = min(lats) - margin, max(lats) + margin
    lon_min, lon_max = min(lons) - margin, max(lons) + margin

    # Define the map projection
    projection = ccrs.PlateCarree()

    # Create the figure and axis with Cartopy projection
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': projection})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.RIVERS)

    # Plot terminal nodes
    for node_id, (lat, lon) in node_coords.items():
        if nodes[node_id].type == "terminal":
            ax.plot(lon, lat, 'bo', markersize=6, transform=ccrs.PlateCarree())
            ax.text(lon + 0.01, lat + 0.01, node_id, fontsize=8, transform=ccrs.PlateCarree())

    # Assign unique colors to each barge
    cmap = cm.get_cmap('tab20', len(barges))
    barge_colors = {barge_id: cmap(i) for i, barge_id in enumerate(barges.keys())}

    # Define edge start and end points for barges in the top-right corner
    num_barges = len(barges)
    edge_points_top = [
        (lat_max, lon_min + (lon_max - lon_min) / 2 + i * (lon_max - lon_min) / (2 * num_barges))
        for i in range(num_barges)
    ]  # Right half of the top edge
    edge_points_right = [
        (lat_min + (lat_max - lat_min) / 2 + i * (lat_max - lat_min) / (2 * num_barges), lon_max)
        for i in range(num_barges)
    ]  # Top half of the right edge
    barge_edges = {barge_id: edge_points_top[i] if i % 2 == 0 else edge_points_right[i]
                   for i, barge_id in enumerate(barges.keys())}

    # Plot barge routes
    x_ijk = variables.get("x_ijk", {})
    for barge_id, routes in x_ijk.items():
        # Skip barges with only one arc
        if sum(1 for (_, _), var in routes.items() if hasattr(var, 'X') and var.X > 0.5) <= 1:
            continue

        # Get the barge's start and end edge points
        edge_point = barge_edges[barge_id]

        # Collect route coordinates
        route_coords = [edge_point]  # Start from the edge
        for (i, j), var in routes.items():
            if hasattr(var, 'X') and var.X > 0.5 and nodes[i].type == "terminal" and nodes[j].type == "terminal":
                route_coords.append(node_coords[i])
                route_coords.append(node_coords[j])
        route_coords.append(edge_point)  # End at the same edge point

        # Remove duplicate consecutive points
        route_coords = [route_coords[0]] + [
            coord for i, coord in enumerate(route_coords[1:]) if coord != route_coords[i]
        ]

        # Extract latitudes and longitudes for plotting
        lats = [coord[0] for coord in route_coords]
        lons = [coord[1] for coord in route_coords]

        # Plot the route
        ax.plot(
            lons, lats,
            color=barge_colors[barge_id],
            linewidth=2,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            label=f"Barge {barge_id}" if f"Barge {barge_id}" not in ax.get_legend_handles_labels()[1] else ""
        )

        # Add arrows to indicate direction
        for idx in range(len(lons) - 1):
            ax.annotate(
                '',
                xy=(lons[idx + 1], lats[idx + 1]),
                xytext=(lons[idx], lats[idx]),
                arrowprops=dict(arrowstyle="->", color=barge_colors[barge_id], lw=1.5),
                transform=ccrs.PlateCarree()
            )

    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title("Barge Routes with Terminals", fontsize=14)

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Map saved as {output_file}")






