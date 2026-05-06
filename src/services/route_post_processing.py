from typing import List, Tuple, Optional
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString

from src.models.route_model import RouteResponseModel, RouteSegment, Coordinate
from src.utils.map import load_osm_network


def get_edge_geometry(
        G: nx.MultiDiGraph,
        u: int,
        v: int
) -> List[Coordinate]:
    """
    Returns the full coordinate list for a graph edge,
    using its LineString geometry if available, or falling
    back to the two endpoint nodes.
    """
    # MultiDiGraph can have multiple edges between two nodes — take the first
    edge_data = G.edges[u, v, 0]

    if 'geometry' in edge_data:
        geom: LineString = edge_data['geometry']
        return [
            Coordinate(lat=lat, lng=lng)
            for lng, lat in geom.coords  # Shapely: (x, y) = (lng, lat)
        ]
    else:
        return [
            Coordinate(lat=G.nodes[u]['y'], lng=G.nodes[u]['x']),
            Coordinate(lat=G.nodes[v]['y'], lng=G.nodes[v]['x']),
        ]


def coord_to_nearest_node(G: nx.MultiDiGraph, coord: Coordinate) -> int:
    """Maps a Coordinate to its nearest OSMnx graph node ID."""
    return ox.nearest_nodes(G, X=coord.lng, Y=coord.lat)


def enrich_segment_coordinates(
        G: nx.MultiDiGraph,
        segment: RouteSegment
) -> List[Coordinate]:
    detailed_coords: List[Coordinate] = []

    for i in range(len(segment.coordinates) - 1):
        start = segment.coordinates[i]
        end = segment.coordinates[i + 1]

        u = coord_to_nearest_node(G, start)
        v = coord_to_nearest_node(G, end)

        if u == v:
            continue

        # If directly connected, use the edge geometry
        if G.has_edge(u, v) or G.has_edge(v, u):
            edge_coords = get_edge_geometry(G, u, v)
        else:
            # Nodes aren't directly connected — find the path between them
            try:
                path_nodes = nx.shortest_path(G, u, v, weight='length')
                edge_coords = []
                for j in range(len(path_nodes) - 1):
                    segment_coords = get_edge_geometry(G, path_nodes[j], path_nodes[j + 1])
                    if edge_coords and edge_coords[-1] == segment_coords[0]:
                        edge_coords.extend(segment_coords[1:])
                    else:
                        edge_coords.extend(segment_coords)
            except nx.NetworkXNoPath:
                # Last resort: straight line
                edge_coords = [
                    Coordinate(lat=G.nodes[u]['y'], lng=G.nodes[u]['x']),
                    Coordinate(lat=G.nodes[v]['y'], lng=G.nodes[v]['x']),
                ]

        if detailed_coords and detailed_coords[-1] == edge_coords[0]:
            detailed_coords.extend(edge_coords[1:])
        else:
            detailed_coords.extend(edge_coords)

    return detailed_coords



def enrich_route_with_geometry(
        route: RouteResponseModel,
        G: nx.MultiDiGraph
) -> RouteResponseModel:
    """
    Post-processes a RouteResponseModel by replacing sparse node-only
    segment coordinates with full road geometries from the OSMnx graph.

    Args:
        route: The original RouteResponseModel.
        G:     The OSMnx graph used during routing.

    Returns:
        A new RouteResponseModel with geometry-enriched segments.
    """
    enriched_segments: List[RouteSegment] = []

    for segment in route.segments:
        detailed_coords = enrich_segment_coordinates(G, segment)

        enriched_segments.append(RouteSegment(
            id=segment.id,
            coordinates=detailed_coords,
            distance_meters=segment.distance_meters,
            travel_time_seconds=segment.travel_time_seconds,
            hazard_score=segment.hazard_score,
        ))

    return RouteResponseModel(
        id=route.id,
        type=route.type,
        rain_intensity=route.rain_intensity,
        depot=route.depot,
        segments=enriched_segments,
        delivery_stops=route.delivery_stops,
        total_distance_meters=route.total_distance_meters,
        total_travel_time_seconds=route.total_travel_time_seconds,
        average_hazard_score=route.average_hazard_score,
    )

def postprocess_route(route: RouteResponseModel) -> RouteResponseModel:
    """
    Takes a RouteResponseModel and returns a new one with geometry-enriched
    segments. Loads the OSMnx graph internally.
    """
    G = load_osm_network()
    return enrich_route_with_geometry(route, G)
