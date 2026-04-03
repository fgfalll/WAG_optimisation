"""
Grid Management for the Unified Physics Engine.

Provides grid abstraction supporting both simple Cartesian grids and
complex corner-point grids used in detailed simulations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from enum import Enum

from ..base.engine_config import GridConfig


class GridType(Enum):
    """Grid geometry type."""
    CARTESIAN = "cartesian"           # Uniform Cartesian grid
    CORNER_POINT = "corner_point"     # Corner-point geometry
    RADIAL = "radial"                 # Radial/cylindrical grid
    UNSTRUCTURED = "unstructured"     # Unstructured/Voronoi grid


@dataclass
class CellGeometry:
    """Geometry data for a single cell."""
    volume: float          # Cell volume (ft^3)
    centroid: Tuple[float, float, float]  # Cell center (x, y, z) in ft
    dimensions: Tuple[float, float, float]  # Cell sizes (dx, dy, dz) in ft

    # Face areas (for flux calculations)
    area_x: float          # Area of face normal to x (ft^2)
    area_y: float          # Area of face normal to y (ft^2)
    area_z: float          # Area of face normal to z (ft^2)

    # Distances to neighboring cell centers
    dist_x: Tuple[float, float]  # (negative, positive) direction distances
    dist_y: Tuple[float, float]
    dist_z: Tuple[float, float]


@dataclass
class NeighborInfo:
    """Information about cell connectivity."""
    neighbors: List[Tuple[int, str]]  # List of (cell_index, face_direction)
    transmissibility: Dict[str, float]  # Transmissibility by face direction


class GridManager(ABC):
    """
    Abstract base class for grid managers.

    Provides unified interface for grid operations regardless of grid type.
    """

    def __init__(self, config: GridConfig):
        """
        Initialize grid manager.

        Args:
            config: Grid configuration.
        """
        self.config = config
        self._n_cells = 0
        self._n_active = 0
        self._active_mask: Optional[np.ndarray] = None
        self._geometry: Optional[List[CellGeometry]] = None
        self._neighbors: Optional[List[NeighborInfo]] = None

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self._n_cells

    @property
    def n_active(self) -> int:
        """Number of active cells."""
        return self._n_active

    @property
    def active_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of active cells."""
        return self._active_mask

    @property
    def grid_type(self) -> GridType:
        """Type of grid geometry."""
        return GridType.CARTESIAN

    @abstractmethod
    def get_cell_index(self, i: int, j: int, k: int) -> int:
        """
        Get linear cell index from (i, j, k) coordinates.

        Args:
            i, j, k: Grid indices.

        Returns:
            Linear cell index.
        """
        pass

    @abstractmethod
    def get_cell_coordinates(self, index: int) -> Tuple[int, int, int]:
        """
        Get (i, j, k) coordinates from linear index.

        Args:
            index: Linear cell index.

        Returns:
            Tuple of (i, j, k) coordinates.
        """
        pass

    @abstractmethod
    def get_cell_geometry(self, index: int) -> CellGeometry:
        """
        Get geometry information for a cell.

        Args:
            index: Cell index.

        Returns:
            CellGeometry with cell dimensions and properties.
        """
        pass

    @abstractmethod
    def get_neighbors(self, index: int) -> NeighborInfo:
        """
        Get neighbor information for a cell.

        Args:
            index: Cell index.

        Returns:
            NeighborInfo with connectivity and transmissibility.
        """
        pass

    @abstractmethod
    def calculate_transmissibility(
        self,
        permeability: np.ndarray,
        cell_i: int,
        cell_j: int,
        direction: str,
    ) -> float:
        """
        Calculate transmissibility between two cells.

        Args:
            permeability: Permeability field (n_cells, 3) or (n_cells,).
            cell_i: First cell index.
            cell_j: Second cell index.
            direction: Direction ('x', 'y', or 'z').

        Returns:
            Transmissibility value.
        """
        pass

    def get_active_indices(self) -> np.ndarray:
        """Get array of active cell indices."""
        if self._active_mask is None:
            return np.arange(self._n_cells)
        return np.where(self._active_mask)[0]

    def get_cell_volume(self, index: int) -> float:
        """Get volume of a specific cell."""
        geom = self.get_cell_geometry(index)
        return geom.volume

    def get_total_volume(self) -> float:
        """Get total volume of active cells."""
        if self._active_mask is None:
            return sum(geom.volume for geom in self._geometry)
        return sum(
            geom.volume
            for i, geom in enumerate(self._geometry)
            if self._active_mask[i]
        )


class CartesianGridManager(GridManager):
    """
    Grid manager for uniform Cartesian grids.

    Simple, regular grids used in fast screening simulations.
    """

    def __init__(self, config: GridConfig):
        """
        Initialize Cartesian grid manager.

        Args:
            config: Grid configuration.
        """
        super().__init__(config)
        self._n_cells = config.nx * config.ny * config.nz
        self._n_active = self._n_cells
        self._active_mask = np.ones(self._n_cells, dtype=bool)
        self._build_geometry()

    @property
    def grid_type(self) -> GridType:
        return GridType.CARTESIAN

    def _build_geometry(self) -> None:
        """Build cell geometry for all cells."""
        self._geometry = []
        self._neighbors = []

        dx, dy, dz = self.config.dx, self.config.dy, self.config.dz
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz

        volume = dx * dy * dz
        area_x = dy * dz
        area_y = dx * dz
        area_z = dx * dy

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Centroid coordinates
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    z = (k + 0.5) * dz

                    # Distances to neighbors
                    dist_x = (dx if i > 0 else dx, dx if i < nx - 1 else dx)
                    dist_y = (dy if j > 0 else dy, dy if j < ny - 1 else dy)
                    dist_z = (dz if k > 0 else dz, dz if k < nz - 1 else dz)

                    geom = CellGeometry(
                        volume=volume,
                        centroid=(x, y, z),
                        dimensions=(dx, dy, dz),
                        area_x=area_x,
                        area_y=area_y,
                        area_z=area_z,
                        dist_x=dist_x,
                        dist_y=dist_y,
                        dist_z=dist_z,
                    )
                    self._geometry.append(geom)

        # Build neighbor lists
        for idx in range(self._n_cells):
            self._neighbors.append(self._find_neighbors(idx))

    def _find_neighbors(self, index: int) -> NeighborInfo:
        """Find neighbors for a cell in Cartesian grid."""
        i, j, k = self.get_cell_coordinates(index)
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz

        neighbors = []
        transmissibility = {}

        # Check each direction
        for di, dj, dk, direction in [
            (-1, 0, 0, "x-"),
            (1, 0, 0, "x+"),
            (0, -1, 0, "y-"),
            (0, 1, 0, "y+"),
            (0, 0, -1, "z-"),
            (0, 0, 1, "z+"),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                n_idx = self.get_cell_index(ni, nj, nk)
                neighbors.append((n_idx, direction))
                # Placeholder transmissibility (actual value calculated with permeability)
                transmissibility[direction] = 1.0

        return NeighborInfo(neighbors=neighbors, transmissibility=transmissibility)

    def get_cell_index(self, i: int, j: int, k: int) -> int:
        return k * self.config.nx * self.config.ny + j * self.config.nx + i

    def get_cell_coordinates(self, index: int) -> Tuple[int, int, int]:
        nx, ny = self.config.nx, self.config.ny
        k = index // (nx * ny)
        j = (index % (nx * ny)) // nx
        i = (index % (nx * ny)) % nx
        return i, j, k

    def get_cell_geometry(self, index: int) -> CellGeometry:
        return self._geometry[index]

    def get_neighbors(self, index: int) -> NeighborInfo:
        return self._neighbors[index]

    def calculate_transmissibility(
        self,
        permeability: np.ndarray,
        cell_i: int,
        cell_j: int,
        direction: str,
    ) -> float:
        """
        Calculate transmissibility between adjacent cells.

        Uses harmonic averaging for permeability at interface.
        """
        # Get cell geometry
        geom_i = self.get_cell_geometry(cell_i)
        geom_j = self.get_cell_geometry(cell_j)

        # Get permeability values
        if permeability.ndim == 1:
            k_i = k_j = permeability[cell_i]
        else:
            dir_idx = {"x": 0, "y": 1, "z": 2}[direction[0]]
            k_i = permeability[cell_i, dir_idx]
            k_j = permeability[cell_j, dir_idx]

        # Calculate transmissibility
        if direction[0] == "x":
            area = geom_i.area_x
            dist = geom_i.dimensions[0]
        elif direction[0] == "y":
            area = geom_i.area_y
            dist = geom_i.dimensions[1]
        else:  # z
            area = geom_i.area_z
            dist = geom_i.dimensions[2]

        # Harmonic average of permeabilities
        k_avg = 2 * k_i * k_j / (k_i + k_j + 1e-20)

        # Conversion factor: 6.33e-3 for field units (mD-ft to ft^3/day/psi)
        transmissibility = 6.33e-3 * k_avg * area / dist

        return transmissibility


class CornerPointGridManager(GridManager):
    """
    Grid manager for corner-point geometry grids.

    Complex grids used in detailed reservoir simulations.
    """

    def __init__(
        self,
        config: GridConfig,
        coord: np.ndarray,  # Corner coordinates (nx+1, ny+1, nz+1, 3)
        zcorn: np.ndarray,  # Depth values (nx+1, ny+1, 2, nz)
        actnum: Optional[np.ndarray] = None,  # Active cell numbers (nx, ny, nz)
    ):
        """
        Initialize corner-point grid manager.

        Args:
            config: Grid configuration.
            coord: Corner point coordinates.
            zcorn: Depth values at corners.
            actnum: Active cell indicator (0=inactive, 1=active).
        """
        super().__init__(config)
        self.coord = coord
        self.zcorn = zcorn
        self.actnum = actnum
        self._n_cells = config.nx * config.ny * config.nz

        if actnum is not None:
            self._active_mask = actnum.flatten().astype(bool)
            self._n_active = int(np.sum(self._active_mask))
        else:
            self._active_mask = np.ones(self._n_cells, dtype=bool)
            self._n_active = self._n_cells

        self._build_geometry()

    @property
    def grid_type(self) -> GridType:
        return GridType.CORNER_POINT

    def _build_geometry(self) -> None:
        """Build cell geometry from corner-point data."""
        # This is a simplified implementation
        # Real corner-point grids require more sophisticated geometry calculation
        self._geometry = []
        self._neighbors = []

        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = self.get_cell_index(i, j, k)

                    if not self._active_mask[idx]:
                        self._geometry.append(None)
                        self._neighbors.append(None)
                        continue

                    # Approximate geometry (simplified)
                    # In reality, would compute from corner-point data
                    dx = self.config.dx
                    dy = self.config.dy
                    dz = self.config.dz

                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    z = (k + 0.5) * dz

                    geom = CellGeometry(
                        volume=dx * dy * dz,
                        centroid=(x, y, z),
                        dimensions=(dx, dy, dz),
                        area_x=dy * dz,
                        area_y=dx * dz,
                        area_z=dx * dy,
                        dist_x=(dx, dx),
                        dist_y=(dy, dy),
                        dist_z=(dz, dz),
                    )
                    self._geometry.append(geom)

        # Build neighbor lists (similar to Cartesian but with active mask)
        for idx in range(self._n_cells):
            if self._active_mask[idx]:
                self._neighbors.append(self._find_neighbors(idx))
            else:
                self._neighbors.append(None)

    def _find_neighbors(self, index: int) -> NeighborInfo:
        """Find active neighbors for a cell."""
        i, j, k = self.get_cell_coordinates(index)
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz

        neighbors = []
        transmissibility = {}

        for di, dj, dk, direction in [
            (-1, 0, 0, "x-"),
            (1, 0, 0, "x+"),
            (0, -1, 0, "y-"),
            (0, 1, 0, "y+"),
            (0, 0, -1, "z-"),
            (0, 0, 1, "z+"),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                n_idx = self.get_cell_index(ni, nj, nk)
                if self._active_mask[n_idx]:
                    neighbors.append((n_idx, direction))
                    transmissibility[direction] = 1.0

        return NeighborInfo(neighbors=neighbors, transmissibility=transmissibility)

    def get_cell_index(self, i: int, j: int, k: int) -> int:
        return k * self.config.nx * self.config.ny + j * self.config.nx + i

    def get_cell_coordinates(self, index: int) -> Tuple[int, int, int]:
        nx, ny = self.config.nx, self.config.ny
        k = index // (nx * ny)
        j = (index % (nx * ny)) // nx
        i = (index % (nx * ny)) % nx
        return i, j, k

    def get_cell_geometry(self, index: int) -> CellGeometry:
        return self._geometry[index]

    def get_neighbors(self, index: int) -> NeighborInfo:
        return self._neighbors[index]

    def calculate_transmissibility(
        self,
        permeability: np.ndarray,
        cell_i: int,
        cell_j: int,
        direction: str,
    ) -> float:
        """Calculate transmissibility for corner-point grid."""
        # Simplified - same as Cartesian
        # Real implementation would account for non-orthogonal faces
        return super().calculate_transmissibility(permeability, cell_i, cell_j, direction)


def create_grid_manager(config: GridConfig) -> GridManager:
    """
    Factory function to create appropriate grid manager.

    Args:
        config: Grid configuration.

    Returns:
        GridManager instance appropriate for the grid type.
    """
    if config.cartesian:
        return CartesianGridManager(config)
    else:
        # Would need corner-point data
        raise NotImplementedError(
            "Corner-point grids require coordinate and depth data"
        )
