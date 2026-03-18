from enum import Enum


class RouteType(str, Enum):
    SAFE = "safe"
    BALANCED = "balanced"
    FAST = "fast"


class RainIntensity(Enum):
    RI1 = "RI1"
    RI2 = "RI2"
    RI3 = "RI3"
    RI4 = "RI4"
    RI5 = "RI5"
