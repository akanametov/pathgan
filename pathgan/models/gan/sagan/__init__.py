"""Self-Attention GAN."""

from .discriminator import MapDiscriminator, PointDiscriminator
from .generator import SAGenerator

__all__ = [
    "MapDiscriminator",
    "PointDiscriminator",
    "SAGenerator",
]
