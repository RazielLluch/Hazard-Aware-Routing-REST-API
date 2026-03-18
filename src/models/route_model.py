from pydantic import field_validator
from .model import Model
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
        return latitude

    @classmethod
    @field_validator("lng")
    def validate_lng(cls, longitude):
        if longitude < -180 or longitude > 180:
            raise ValueError("Longitude must be between -180 and 180")
        return longitude


class RouteSegment(Model):
    id: Optional[str] = None
    coordinates: List[Coordinate]
    distance_meters: Optional[float] = None
    travel_time_seconds: Optional[float] = None
    hazard_score: Optional[float] = None


class DeliveryStop(Model):
    id: Optional[str] = None
    location: Coordinate
    sequence: Optional[int] = None
    label: Optional[str] = None

    @classmethod
    @field_validator("sequence")
    def validate_sequence(cls, sequence):
        if sequence < 1:
            raise ValueError("Sequence must be greater than 1")
        return sequence


class RouteRequestModel(Model):
    rain_intensity: RainIntensity = RainIntensity.RI1
    route_type: RouteType = RouteType.BALANCED
    depot: DeliveryStop
    delivery_stops: List[DeliveryStop]

    @classmethod
    @field_validator("delivery_stops")
    def validate_delivery_stops(cls, stops):
        if len(stops) < 2:
            raise ValueError("Delivery stops must have at least 2 items")
        count = sum(1 for stop in stops if stop.sequence == 1)
        if count > 1:
            raise ValueError("Cannot have multiple depots(delivery stops with sequence number 1)")
        elif count < 1:
            raise ValueError("Cannot have no depots(0 delivery stops)")
        return stops


class RouteResponseModel(Model):
    id: UUID
    rain_intensity: RainIntensity
    type: RouteType
    depot: DeliveryStop
    segments: List[RouteSegment]
    delivery_stops: List[DeliveryStop]

    total_distance_meters: Optional[float] = None
    total_travel_time_seconds: Optional[float] = None
    average_hazard_score: Optional[float] = None

    @classmethod
    @field_validator("segments")
    def validate_segments(cls, segments: List[RouteSegment]):
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]

            prev_end = prev.coordinates[-1]
            curr_start = curr.coordinates[0]

            if prev_end != curr_start:
                raise ValueError("Route segments are not continuous")

        return segments

