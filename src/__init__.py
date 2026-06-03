"""boiling-phasefield-3d  —  phase-field simulation of boiling heat transfer."""
from .params      import SimParams
from .solver      import run_2d, run_3d
from .diagnostics import bubble_radius_2d, bubble_radius_3d, interface_position_1d
