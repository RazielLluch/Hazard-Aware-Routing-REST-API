from enum import Enum


class RoutingProfile(str, Enum):
    SAFE = "safe"
    BALANCED = "balanced"
    FAST = "fast"


class RainIntensity(Enum):
    RI1 = 1
    RI2 = 2
    RI3 = 3
    RI4 = 4
    RI5 = 5
