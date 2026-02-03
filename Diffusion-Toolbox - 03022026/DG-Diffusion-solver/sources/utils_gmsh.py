"""
OneDSampleMeshGenerator
-----------------------

A tiny helper around the Gmsh Python API that creates a 1D line mesh
for diffusion simulations. It tags the two boundary points
("upstream", "downstream") and the interior line ("sample") as
**physical groups**, so downstream solvers (FEniCSx, etc.) can pick
them up by name.

Requirements
~~~~~~~~~~~~
- gmsh >= 4.11  (pip install gmsh)
- matplotlib    (pip install matplotlib)  # only for plot_mesh()

Usage
~~~~~
>>> gen = OneDSampleMeshGenerator(length=0.005,
...                               mode="Use Divisions",
...                               num_divisions=20,
...                               output_dir="./gmsh")
>>> msh_path = gen.generate_mesh()
>>> gen.plot_mesh()   # optional visualization
"""

from __future__ import annotations

import os
from typing import Tuple

import gmsh
import matplotlib.pyplot as plt


class OneDSampleMeshGenerator:
    """
    Create a 1D line segment mesh with physical group tags using Gmsh.

    Parameters
    ----------
    length : float, optional
        Length of the 1D sample (meters). Default is 0.005 (5 mm).
    mode : {"Use Mesh Size", "Use Divisions"}, optional
        Meshing strategy:
          - "Use Mesh Size": target a uniform element size along the line.
          - "Use Divisions": split the line into a fixed number of segments.
        Default is "Use Mesh Size".
    mesh_size : float, optional
        Target element size (meters) when `mode="Use Mesh Size"`. Default 3e-4.
    num_divisions : int, optional
        Number of equal segments when `mode="Use Divisions"`. Default 17.
    output_dir : str, optional
        Directory where the generated `.msh` file will be written. Default "./gmsh".
    """

    # Recommended tags (kept stable so downstream code can rely on them)
    TAG_UPSTREAM = 3
    TAG_DOWNSTREAM = 4
    TAG_SAMPLE = 5

    def __init__(
        self,
        length: float = 0.005,
        mode: str = "Use Mesh Size",
        mesh_size: float = 0.0003,
        num_divisions: int = 17,
        output_dir: str = "./gmsh",
    ) -> None:
        self.length = float(length)
        self.mode = str(mode)
        self.mesh_size = float(mesh_size)
        self.num_divisions = int(num_divisions)
        self.output_dir = str(output_dir)
        self.mesh_filename = os.path.join(self.output_dir, "sample_1D.msh")

        if self.length <= 0:
            raise ValueError("`length` must be positive.")
        if self.mode not in {"Use Mesh Size", "Use Divisions"}:
            raise ValueError("`mode` must be 'Use Mesh Size' or 'Use Divisions'.")
        if self.mode == "Use Mesh Size" and self.mesh_size <= 0:
            raise ValueError("`mesh_size` must be positive when mode='Use Mesh Size'.")
        if self.mode == "Use Divisions" and self.num_divisions < 1:
            raise ValueError("`num_divisions` must be >= 1 when mode='Use Divisions'.")

    # ------------------------------
    # Public API
    # ------------------------------
    def generate_mesh(self) -> str:
        """
        Build the 1D mesh and write it to disk.

        Returns
        -------
        str
            Absolute path to the written `.msh` file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        gmsh.initialize()
        try:
            gmsh.model.add("1D_diffusion_sample")

            # --- Geometry: two points and a connecting line
            if self.mode == "Use Mesh Size":
                # Directly use the target size on both end points
                p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.mesh_size)
                p2 = gmsh.model.geo.addPoint(self.length, 0.0, 0.0, self.mesh_size)
            else:
                # Estimate a characteristic length from the number of divisions
                h = self.length / self.num_divisions
                p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)
                p2 = gmsh.model.geo.addPoint(self.length, 0.0, 0.0, h)

            line = gmsh.model.geo.addLine(p1, p2)

            # If we mesh by fixed divisions, set transfinite on the curve
            if self.mode == "Use Divisions":
                # +1 nodes means exactly `num_divisions` line elements
                gmsh.model.geo.mesh.setTransfiniteCurve(line, self.num_divisions + 1)

            # IMPORTANT: promote CAD entities to the "model" before adding physical groups
            gmsh.model.geo.synchronize()

            # --- Physical groups (stable tags + human-readable names)
            gmsh.model.addPhysicalGroup(0, [p1], tag=self.TAG_UPSTREAM)
            gmsh.model.setPhysicalName(0, self.TAG_UPSTREAM, "upstream")

            gmsh.model.addPhysicalGroup(0, [p2], tag=self.TAG_DOWNSTREAM)
            gmsh.model.setPhysicalName(0, self.TAG_DOWNSTREAM, "downstream")

            gmsh.model.addPhysicalGroup(1, [line], tag=self.TAG_SAMPLE)
            gmsh.model.setPhysicalName(1, self.TAG_SAMPLE, "sample")

            # --- Generate the 1D mesh and write it out
            gmsh.model.mesh.generate(1)
            gmsh.write(self.mesh_filename)

        finally:
            # Always finalize to free Gmsh state, even on error
            gmsh.finalize()

        print(f"✅ Mesh saved to: {os.path.abspath(self.mesh_filename)}")
        return os.path.abspath(self.mesh_filename)

    def plot_mesh(self) -> None:
        """
        Visualize the mesh and its physical groups using matplotlib.
        Enhanced for a professional, publication-ready look.

        Notes
        -----
        - This is a lightweight plot just for inspection (not an FE plot).
        - Requires `matplotlib`.
        """
        if not os.path.isfile(self.mesh_filename):
            raise FileNotFoundError(
                f"Mesh file not found: {self.mesh_filename}. "
                "Call `generate_mesh()` first."
            )

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress all "Info" messages
        gmsh.model.add("1D_diffusion_sample")
        try:
            gmsh.open(self.mesh_filename)

            # ---- Nodes and elements
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3)

            # Get first 1D element block (usually 2-node lines)
            etypes, _, conn = gmsh.model.mesh.getElements(1)
            if not etypes:
                raise RuntimeError("No 1D elements found in the mesh.")
            
            line_type = int(etypes[0])
            
            # getElementProperties returns a tuple; index 3 is the number of nodes per element
            props = gmsh.model.mesh.getElementProperties(line_type)
            nper = int(props[3])  # number of nodes per 1D element (usually 2 for linear)
            
            # Some meshes may come in multiple 1D blocks; we use the first block here
            line_conn = conn[0].reshape(-1, nper)

            # ---- Plotting Setup (Professional Style)
            fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
            
            # Remove spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['bottom'].set_linewidth(1.2)
            
            # ---- Plot Segments (The Mesh)
            # We plot the full line first as a base
            ax.plot([0, self.length], [0, 0], color='#ecf0f1', linewidth=6, zorder=0)
            
            # Plot individual elements to show mesh density
            for i, e in enumerate(line_conn):
                # For higher-order lines we only connect the first and last node visually
                a, b = int(e[0]) - 1, int(e[-1]) - 1
                x = [node_coords[a, 0], node_coords[b, 0]]
                
                # Plot element body
                ax.plot(x, [0.0, 0.0], color='#34495e', linewidth=2, zorder=1)
                
                # Plot nodes as ticks to visualize discretization
                # Mark the start of the element
                ax.plot(x[0], 0, marker='|', color='#2c3e50', markersize=10, markeredgewidth=1.5, zorder=2)
                # Mark the end of the last element
                if i == len(line_conn) - 1:
                    ax.plot(x[1], 0, marker='|', color='#2c3e50', markersize=10, markeredgewidth=1.5, zorder=2)

            # ---- Annotate physical groups
            # Dictionary to store group info for legends/labels
            # Groups: {tag: {'name': name, 'x': center_x, 'dim': dim}}
            
            for dim, ptag in gmsh.model.getPhysicalGroups():
                name = gmsh.model.getPhysicalName(dim, ptag) or f"group_{ptag}"
                
                # Find entities and center
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
                
                # Calculate a representative position (naive avg of bounding box)
                min_x_list, max_x_list = [], []
                for ent in entities:
                    xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(dim, ent)
                    min_x_list.append(xmin)
                    max_x_list.append(xmax)
                
                if not min_x_list: continue
                
                mid_x = 0.5 * (min(min_x_list) + max(max_x_list))
                
                if dim == 0:  # Points (Boundaries)
                    # High-contrast markers for BCs
                    color = '#e74c3c' if 'upstream' in name.lower() else '#3498db'
                    ax.scatter([mid_x], [0.0], s=120, color=color, edgecolors='white', linewidth=1.5, zorder=10, label=f"{name} (ID={ptag})")
                    
                    # Annotate below axis
                    ax.annotate(f"{name}\n(ID={ptag})", 
                                xy=(mid_x, 0), xytext=(mid_x, -0.05),
                                ha='center', va='top', fontsize=9, fontweight='bold', color=color,
                                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5))

                elif dim == 1:  # Sample Line
                    # Annotate above axis
                    ax.text(mid_x, 0.03, f"{name.upper()} DOMAIN (ID={ptag})", 
                            ha="center", va="bottom", fontsize=10, fontweight='bold', color='#34495e',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdc3c7", alpha=0.9))
                    
                    # Add dimension arrow
                    ax.annotate("", xy=(0, 0.015), xytext=(self.length, 0.015),
                                arrowprops=dict(arrowstyle="<->", color='#7f8c8d', lw=1))
                    ax.text(mid_x, 0.018, f"L = {self.length*1000:.2f} mm", ha='center', va='bottom', fontsize=9, color='#7f8c8d')

            # ---- Final Polish
            ax.set_title("1D Finite Element Mesh & Boundary Tags", fontsize=12, pad=20, fontweight='bold', loc='left')
            
            # X-Axis formatting
            ax.set_xlabel("Position (m)", fontsize=10, labelpad=10)
            ax.tick_params(axis='x', colors='#333333', labelsize=9)
            
            # Remove Y-axis completely
            ax.set_yticks([])
            
            # Add padding to view
            padding = self.length * 0.1
            ax.set_xlim(-padding, self.length + padding)
            # Set Y limits to center the 1D line
            ax.set_ylim(-0.1, 0.1)
            
            plt.tight_layout()
            plt.show()

        finally:
            gmsh.finalize()