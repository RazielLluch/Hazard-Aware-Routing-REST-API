from pydantic import field_validator
from model import Model
from uuid import UUID
from typing import List, Optional
from ..schemas.enums import RouteType, RainIntensity


class Coordinate(Model):
    lat: float
    lng: float

    @classmethod
    @field_validator("lat")
    def validate_lat(cls, latitude):
        if latitude < -90 or latitude > 90:
            raise ValueError("Latitude must be between -90 and 90")

    @classmethod
    @field_validator("lng")
    def validate_lng(cls, longitude):
        if longitude < -180 or longitude > 180:
            raise ValueError("Longitude must be between -180 and 180")


class RouteSegment(Model):
    id: str
    coordinates: List[Coordinate]
    distance_meters: Optional[float] = None
    travel_time_seconds: Optional[float] = None
    hazard_score: Optional[float] = None


class DeliveryStop(Model):
    id: str
    location: Coordinate
    sequence: Optional[int] = None
    label: Optional[str] = None

    @classmethod
    @field_validator("sequence")
    def validate_sequence(cls, sequence):
        if sequence < 1:
            raise ValueError("Sequence must be greater than 1")


class RouteRequestModel(Model):
    id: UUID
    delivery_stops: List[DeliveryStop]
    rain_intensity: RainIntensity
    route_type: RouteType

    @classmethod
    @field_validator("delivery_stops")
    def validate_delivery_stops(cls, stops):
        count = sum(1 for stop in stops if stops.sequence == 1)
        if count > 1:
            raise ValueError("Cannot have multiple depots(delivery stops with sequence number 1)")
        elif count < 1:
            raise ValueError("Cannot have no depots(0 delivery stops)")


class RouteResponseModel(Model):
    id: UUID
    type: RouteType
    segments: List[RouteSegment]

    depot: Coordinate
    delivery_stops: List[DeliveryStop]

    total_distance_meters: Optional[float] = None
    total_travel_time_seconds: Optional[float] = None
    average_hazard_score: Optional[float] = None

    @classmethod
    @field_validator("segments")
    def validate_segments(cls, segments: List[RouteSegment]):
        for i, segment in enumerate(segments):
            if i == 0:
                continue  # no previous element

            prev = segments[i - 1]

            if prev.sequence != segment.sequence:
                raise ValueError("Route node mismatch!")


