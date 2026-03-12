from typing import List

import osmnx as ox
import networkx as nx
import random
from src.models.route_model import DeliveryStop, Coordinate, RouteSegment

LOCATION = "La Trinidad, Benguet, Philippines"


def nearest_edge_node(G, lng, lat):
    """
    Find the closest road edge and return the closest node of that edge.
    """
    u, v, key = ox.distance.nearest_edges(G, lng, lat)

    dist_u = ox.distance.great_circle(
        lat, lng,
        G.nodes[u]["y"], G.nodes[u]["x"]
    )

    dist_v = ox.distance.great_circle(
        lat, lng,
        G.nodes[v]["y"], G.nodes[v]["x"]
    )

    return u if dist_u < dist_v else v


def route_to_coordinates(G, route):
    """
    Convert a node route to detailed coordinates using edge geometry.
    """
    coords = []

    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)[0]

        if "geometry" in edge_data:
            xs, ys = edge_data["geometry"].xy
            for x, y in zip(xs, ys):
                coords.append({"lat": y, "lng": x})
        else:
            coords.append({
                "lat": G.nodes[u]["y"],
                "lng": G.nodes[u]["x"]
            })

    return coords


def load_osm_network(location=LOCATION, network_type='all'):
    """
    Load OSM road network using OSMNX.

    Parameters:
    -----------
    network_type : str
        Type of street network ('drive', 'walk', 'bike', 'all')
    """

    try:
        G = ox.load_graphml(filepath="../../public/data/la_trinidad.graphml")
        print("Loaded La Trinidad, Benguet graphml file")
    except FileNotFoundError:
        print(f"Loading OSM network for {location}...")
        G = ox.graph_from_place(location, network_type=network_type, which_result=1, retain_all=False, simplify=True)
        G = ox.add_edge_bearings(G)
        # Ensure CRS is WGS84 for geographic coordinates
        G = ox.project_graph(G, to_crs='EPSG:4326')

        ox.save_graphml(G, filepath="../../public/data/la_trinidad.graphml")

        print(f"Loaded {len(G.nodes)} nodes and {len(G.edges)} edges")

        # Get bounding box for reference
        # nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        # osm_bounds = nodes_gdf.total_bounds
        # print(f"OSM network bounds: {osm_bounds}")

    return G


def process_stops(stops: List[DeliveryStop]) -> List[DeliveryStop]:

    G = load_osm_network()

    processed_stops = []

    for i, stop in enumerate(stops):
        node_id = nearest_edge_node(G, stop.location.lng, stop.location.lat)
        node = G.nodes[node_id]

        print("Delivery Stop: ", stop)
        print("Node: ", node)

        coordinate = Coordinate(lng=node['x'], lat=node['y'])
        delivery_stop = DeliveryStop(
            location=coordinate,
            sequence=1 if i == 0 else None
        )

        processed_stops.append(delivery_stop)

    print("PROCESSED STOPS: ", processed_stops)

    return processed_stops


def randomize_delivery_stops(stops: List[DeliveryStop]):

    sequence = list(range(2, len(stops)+1))

    random.shuffle(sequence)

    for i in range(1, len(stops)):
        stops[i].sequence = sequence[i-1]

    return stops


def generate_segment(stop1: DeliveryStop, stop2: DeliveryStop):

    G = load_osm_network()

    node1 = nearest_edge_node(G, stop1.location.lng, stop1.location.lat)
    node2 = nearest_edge_node(G, stop2.location.lng, stop2.location.lat)

    route = nx.shortest_path(G, node1, node2, weight="length", method="bellman-ford")

    coordinates: List[Coordinate] = [
        Coordinate(lat=G.nodes[n]["y"], lng=G.nodes[n]["x"])
        for n in route
    ]

    segment = RouteSegment(
        coordinates=coordinates
    )

    print("SEGMENT: ", segment)

    return segment
