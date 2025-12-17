"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Drifting Environment - A racing track environment for drift simulation.
This module provides a simple racing track with left and right boundaries
for collision checking. Supports straight, oval, and L-shaped track types.

@required-scripts: None (standalone module)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


class DriftingEnv:
    """
    A racing track environment for drift simulation.
    
    The track is defined as a simple oval/rectangular racing track with
    configurable dimensions. Boundaries are computed for collision checking.
    """
    
    def __init__(self, track_type='straight', track_width=8.0, track_length=100.0):
        """
        Initialize the drifting environment.
        
        Args:
            track_type: Type of track ('straight', 'oval', 'l_shape')
            track_width: Width of the track in meters
            track_length: Length of the track in meters
        """
        self.track_type = track_type
        self.track_width = track_width
        self.track_length = track_length
        
        # Generate track boundaries
        self.left_boundary = None
        self.right_boundary = None
        self.centerline = None
        
        self._generate_track()
        
        # Colors for visualization (similar to MATLAB style)
        self.road_color = np.array([150, 150, 150]) / 255  # Gray asphalt
        self.grass_color = np.array([100, 180, 100]) / 255  # Green grass
        self.line_color = 'white'
        self.center_line_color = 'yellow'
        
        # Plot handles
        self.ax = None
        self.road_patch = None
        self.left_boundary_line = None
        self.right_boundary_line = None
        self.center_line = None
        
    def _generate_track(self):
        """Generate track boundaries based on track type."""
        if self.track_type == 'straight':
            self._generate_straight_track()
        elif self.track_type == 'oval':
            self._generate_oval_track()
        elif self.track_type == 'l_shape':
            self._generate_l_shape_track()
        else:
            raise ValueError(f"Unknown track type: {self.track_type}")
    
    def _generate_straight_track(self):
        """Generate a straight track."""
        # Centerline points
        n_points = 100
        x = np.linspace(0, self.track_length, n_points)
        y = np.zeros(n_points)
        
        self.centerline = np.column_stack([x, y])
        
        # Compute boundaries (perpendicular to centerline)
        half_width = self.track_width / 2
        self.left_boundary = np.column_stack([x, y + half_width])
        self.right_boundary = np.column_stack([x, y - half_width])
        
        # Track bounds for plotting
        self.x_min = -5
        self.x_max = self.track_length + 5
        self.y_min = -self.track_width - 5
        self.y_max = self.track_width + 5
        
    def _generate_oval_track(self):
        """Generate an oval track with gentler curves."""
        n_points = 200
        
        # Oval parameters - make semi-minor axis larger for gentler turns
        a = self.track_length / 2  # Semi-major axis
        b = self.track_length / 2.5  # Semi-minor axis
        
        # Parametric oval
        t = np.linspace(0, 2 * np.pi, n_points)
        x = a * np.cos(t) + a
        y = b * np.sin(t)
        
        self.centerline = np.column_stack([x, y])
        
        # Compute boundaries using normal vectors
        half_width = self.track_width / 2
        
        # Compute tangent and normal vectors
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length  # Normal x
        ny = dx / length   # Normal y
        
        self.left_boundary = np.column_stack([x + half_width * nx, y + half_width * ny])
        self.right_boundary = np.column_stack([x - half_width * nx, y - half_width * ny])
        
        # Track bounds
        self.x_min = -10
        self.x_max = 2 * a + 10
        self.y_min = -b - self.track_width - 5
        self.y_max = b + self.track_width + 5
        
    def _generate_l_shape_track(self):
        """Generate an L-shaped track."""
        half_width = self.track_width / 2
        
        # L-shape centerline (two segments)
        seg1_length = self.track_length * 0.6
        seg2_length = self.track_length * 0.4
        
        # First segment (horizontal)
        n1 = 60
        x1 = np.linspace(0, seg1_length, n1)
        y1 = np.zeros(n1)
        
        # Corner (arc)
        n_corner = 20
        corner_radius = self.track_width
        theta = np.linspace(-np.pi/2, 0, n_corner)
        x_corner = seg1_length + corner_radius + corner_radius * np.cos(theta)
        y_corner = corner_radius + corner_radius * np.sin(theta)
        
        # Second segment (vertical)
        n2 = 40
        x2 = np.full(n2, seg1_length + corner_radius)
        y2 = np.linspace(corner_radius, corner_radius + seg2_length, n2)
        
        # Combine
        x = np.concatenate([x1, x_corner, x2])
        y = np.concatenate([y1, y_corner, y2])
        
        self.centerline = np.column_stack([x, y])
        
        # Compute boundaries
        dx = np.gradient(x)
        dy = np.gradient(y)
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length
        
        self.left_boundary = np.column_stack([x + half_width * nx, y + half_width * ny])
        self.right_boundary = np.column_stack([x - half_width * nx, y - half_width * ny])
        
        # Track bounds
        self.x_min = -5
        self.x_max = seg1_length + 2 * corner_radius + 5
        self.y_min = -self.track_width - 5
        self.y_max = corner_radius + seg2_length + 5
    
    def setup_plot(self, ax=None, fig=None):
        """
        Setup the plot with track visualization.
        
        Args:
            ax: Matplotlib axis (optional)
            fig: Matplotlib figure (optional)
            
        Returns:
            ax, fig: The axis and figure handles
        """
        if fig is None:
            fig = plt.figure(figsize=(14, 6))
        if ax is None:
            ax = fig.add_subplot(111)
            
        self.ax = ax
        self.fig = fig
        
        # Draw grass background
        ax.set_facecolor(self.grass_color)
        
        # Draw road surface
        road_vertices = np.vstack([self.left_boundary, self.right_boundary[::-1]])
        road_polygon = MplPolygon(road_vertices, closed=True, 
                                   facecolor=self.road_color, edgecolor='none')
        ax.add_patch(road_polygon)
        self.road_patch = road_polygon
        
        # Draw boundaries
        self.left_boundary_line, = ax.plot(
            self.left_boundary[:, 0], self.left_boundary[:, 1],
            color=self.line_color, linewidth=3, solid_capstyle='round'
        )
        self.right_boundary_line, = ax.plot(
            self.right_boundary[:, 0], self.right_boundary[:, 1],
            color=self.line_color, linewidth=3, solid_capstyle='round'
        )
        
        # Draw center line (dashed yellow)
        self.center_line, = ax.plot(
            self.centerline[:, 0], self.centerline[:, 1],
            color=self.center_line_color, linewidth=2, linestyle='--'
        )
        
        # Set axis properties
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        
        return ax, fig
    
    def check_collision(self, position, robot_radius=0.0):
        """
        Check if a position collides with track boundaries.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            bool: True if collision detected
        """
        x, y = position[0], position[1]
        
        # Find closest point on centerline
        distances = np.linalg.norm(self.centerline - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # Get corresponding boundary points
        left_pt = self.left_boundary[closest_idx]
        right_pt = self.right_boundary[closest_idx]
        center_pt = self.centerline[closest_idx]
        
        # Compute distance from center to boundary (half track width)
        half_width = np.linalg.norm(left_pt - center_pt)
        
        # Check if position is outside track
        dist_from_center = np.linalg.norm(np.array([x, y]) - center_pt)
        
        if dist_from_center + robot_radius > half_width:
            return True
        
        return False
    
    def check_collision_detailed(self, position, robot_radius=0.0):
        """
        Detailed collision check returning which boundary was hit.
        
        Args:
            position: [x, y] position to check
            robot_radius: Radius of the robot for collision margin
            
        Returns:
            dict: {'collision': bool, 'boundary': 'left'/'right'/None, 'distance': float}
        """
        x, y = position[0], position[1]
        
        # Find closest points on boundaries
        dist_to_left = np.min(np.linalg.norm(self.left_boundary - np.array([x, y]), axis=1))
        dist_to_right = np.min(np.linalg.norm(self.right_boundary - np.array([x, y]), axis=1))
        
        # Find closest point on centerline for local track direction
        distances = np.linalg.norm(self.centerline - np.array([x, y]), axis=1)
        closest_idx = np.argmin(distances)
        
        # Get local track coordinate
        left_pt = self.left_boundary[closest_idx]
        right_pt = self.right_boundary[closest_idx]
        center_pt = self.centerline[closest_idx]
        
        # Vector from right to left boundary (perpendicular to track)
        track_normal = left_pt - right_pt
        track_normal = track_normal / np.linalg.norm(track_normal)
        
        # Vector from center to position
        pos_vec = np.array([x, y]) - center_pt
        
        # Signed distance (positive = towards left, negative = towards right)
        signed_dist = np.dot(pos_vec, track_normal)
        half_width = self.track_width / 2
        
        result = {
            'collision': False,
            'boundary': None,
            'distance': min(dist_to_left, dist_to_right),
            'signed_distance': signed_dist
        }
        
        if signed_dist > half_width - robot_radius:
            result['collision'] = True
            result['boundary'] = 'left'
        elif signed_dist < -(half_width - robot_radius):
            result['collision'] = True
            result['boundary'] = 'right'
            
        return result
    
    def get_track_bounds(self):
        """Return the track boundary data for external use."""
        return {
            'left_boundary': self.left_boundary.copy(),
            'right_boundary': self.right_boundary.copy(),
            'centerline': self.centerline.copy(),
            'track_width': self.track_width
        }

