from geopy.distance import geodesic
north, south, east, west = 51.95, 51.90, 4.55, 4.40  # Degrees

# Define the corners of the bounding box
top_left = (north, west)
top_right = (north, east)
bottom_left = (south, west)
bottom_right = (south, east)

# Calculate distances
vertical_distance = geodesic(top_left, bottom_left).kilometers
horizontal_distance = geodesic(top_left, top_right).kilometers

print(f"Vertical Distance: {vertical_distance} km")
print(f"Horizontal Distance: {horizontal_distance} km")

