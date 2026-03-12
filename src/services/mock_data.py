from typing import List

from src.models.route_model import RouteResponseModel, RouteRequestModel, RouteSegment
from ..utils.map import load_osm_network, LOCATION, nearest_edge_node, route_to_coordinates, randomize_delivery_stops, \
    generate_segment, process_stops
from uuid import uuid4
import networkx as nx


def generate_route():
    G = load_osm_network()

    # (lon, lat)
    orig_coords = (120.58012, 16.45382)
    dest_coords = (120.5984, 16.46613)

    # snap to nearest edge nodes
    orig_node = nearest_edge_node(G, *orig_coords)
    dest_node = nearest_edge_node(G, *dest_coords)

    # compute shortest path
    route = nx.shortest_path(G, orig_node, dest_node, weight="length", method="bellman-ford")

    # extract detailed geometry
    coords = route_to_coordinates(G, route)

    # ensure visualization starts/ends at exact input points
    coords.insert(0, {"lat": orig_coords[1], "lng": orig_coords[0]})
    coords.append({"lat": dest_coords[1], "lng": dest_coords[0]})

    return coords


def generate_mock_routes(request: RouteRequestModel):

    print("DEPOT SEQUENCE: ", request.delivery_stops[0].sequence)
    print("STOPS: ", request.delivery_stops)

    processed_stops = process_stops(request.delivery_stops)

    stops = randomize_delivery_stops(processed_stops)
    print("PROCESSED STOPS: ", stops)
    print("PROCESSED DEPOT SEQUENCE: ", stops[0].sequence)
    stops = sorted(stops, key=lambda obj: obj.sequence)

    processed_segments: List[RouteSegment] = []

    for i in range(1, len(stops)):
        processed_segments.append(generate_segment(stops[i-1], stops[i]))

    print("processed_segments: ", processed_segments)

    routes = RouteResponseModel(
        id=uuid4(),
        rain_intensity=request.rain_intensity,
        routing_profile=request.routing_profile,
        segments=processed_segments,
        delivery_stops=stops,
        total_distance_meters=0,
        total_travel_time_seconds=0,
        average_hazard_score=0
    )

    return routes


if __name__ == "__main__":
    route_coords = generate_route()

    print("Route coordinates:", route_coords)
