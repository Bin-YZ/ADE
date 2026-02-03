# === One cell to create a reusable class-based module and launch the UI ===
# It writes mesh_ui.py to disk, then imports and runs MeshGUI().show()

import sys
import os
import re
import time
import traceback
import warnings
import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox, HTML, GridBox
from IPython.display import display, clear_output

# Project deps (adjust search path if needed)
if "../sources" not in sys.path:
    sys.path.append("../sources")

# Try importing backend; if missing, create dummy for UI demonstration
try:
    from utils_gmsh import OneDSampleMeshGenerator
except ImportError:
    class OneDSampleMeshGenerator:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def generate_mesh(self): time.sleep(0.5); print("Simulating mesh generation...")
        def plot_mesh(self): 
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 1))
            plt.title("Preview (Mock)")
            plt.plot([0, 1], [0, 0], '|-k')
            plt.yticks([])
            plt.show()

class MeshGUI:
    # --- STYLE DEFINITIONS (Identical to DiffusionGUI for consistency) ---
    STYLE_CSS = """
    <style>
        .app-container { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f9; padding: 20px; border-radius: 12px; }
        .card { background-color: #ffffff; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; border: 1px solid #e1e4e8; }
        .card-header { font-size: 16px; font-weight: 600; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-bottom: 15px; letter-spacing: 0.5px; }
        .section-label { font-size: 13px; font-weight: bold; color: #7f8c8d; margin-bottom: 5px; text-transform: uppercase; }
        .main-title { font-size: 24px; font-weight: 700; color: #2c3e50; margin: 0; }
        .sub-title { font-size: 14px; color: #7f8c8d; margin-top: 5px; }
        .console-box { background-color: #2d3436; color: #00cec9; font-family: 'Consolas', monospace; padding: 15px; border-radius: 8px; font-size: 12px; }
        .input-hint { font-size: 11px; color: #95a5a6; margin-top: -3px; margin-bottom: 8px; }
    </style>
    """

    def __init__(self, output_dir: str = "./gmsh", default_mesh_basename: str = "sample_1D"):
        self.output_dir = output_dir
        self.default_mesh_basename = default_mesh_basename
        
        self._setup_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Initialize state
        self._update_visibility()

    def _create_widget(self, w_type, description, value, **kwargs):
        kwargs.setdefault('style', {'description_width': 'initial'})
        kwargs.setdefault('layout', Layout(width='98%')) 
        return w_type(value=value, description=description, **kwargs)

    def _setup_widgets(self):
        # Styled widgets
        self.widgets = {
            # --- Geometry ---
            'length': self._create_widget(widgets.BoundedFloatText, "Length (m)", 0.005, min=1e-12, step=0.0001),
            
            # --- Discretization Strategy ---
            'mode': widgets.ToggleButtons(
                options=[("By Divisions", "div"), ("By Mesh Size", "size")],
                value="div",
                style={'button_width': '140px', 'font_weight': 'bold'},
                layout=Layout(width='auto', margin='0 0 10px 0')
            ),
            'mesh_size': self._create_widget(widgets.BoundedFloatText, "Target Size (m)", 0.0003, min=1e-12, step=0.0001),
            'num_div': self._create_widget(widgets.BoundedIntText, "Count", 20, min=2),
            
            # --- Output Config ---
            'filename': widgets.Text(
                value=self.default_mesh_basename, 
                placeholder="e.g. sample_1D",
                description="Filename",
                style={'description_width': 'initial'},
                layout=Layout(width='98%')
            ),
            'overwrite': widgets.Checkbox(value=False, description="Allow Overwrite", indent=False),
            
            # --- Actions ---
            'generate_btn': widgets.Button(
                description="GENERATE MESH", 
                button_style='primary', #  for creation
                layout=Layout(width='100%', height='50px', font_weight='bold', margin='10px 0'),
                icon='magic'
            ),
            'reset_btn': widgets.Button(description="Reset Defaults", icon='refresh', layout=Layout(width='auto')),
            
            # --- Outputs ---
            'log_output': widgets.Output(layout=Layout(width='100%', height='120px', overflow='auto')),
            'plot_output': widgets.Output(layout=Layout(width='100%', min_height='250px')),
        }

    def _setup_layout(self):
        w = self.widgets

        # 1. Header
        header = HTML(f"""
        {self.STYLE_CSS}
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 class="main-title">GMSH <span style="color:#3498db">Generator</span></h1>
                <div class="sub-title">1D Finite Element Mesh Construction</div>
            </div>
            <div style="text-align: right; color: #95a5a6; font-size: 12px;">
                v1.2 | Mesh Engine
            </div>
        </div>
        """)

        # 2. Left Column: Configuration
        
        # Card A: Geometry
        card_geo = VBox([
            HTML("<div class='card-header'>Geometry</div>"),
            HTML("<div class='section-label'>Domain Size</div>"),
            w['length'],
            HTML("<div class='input-hint'>Total length of the 1D domain in meters.</div>")
        ], layout=Layout(classes='card'))

        # Card B: Discretization
        # Use a container to swap inputs based on mode
        self.discretization_container = VBox([
             w['num_div'], w['mesh_size']
        ])
        
        card_mesh = VBox([
            HTML("<div class='card-header'>Discretization</div>"),
            HTML("<div class='section-label'>Strategy</div>"),
            w['mode'],
            HTML("<div class='section-label'>Parameters</div>"),
            self.discretization_container
        ], layout=Layout(classes='card'))

        # Card C: Output Settings
        card_file = VBox([
            HTML("<div class='card-header'>Output Settings</div>"),
            w['filename'],
            HTML(f"<div class='input-hint'>Saved to: {self.output_dir}/...</div>"),
            w['overwrite']
        ], layout=Layout(classes='card'))

        left_col = VBox([card_geo, card_mesh, card_file], layout=Layout(width='40%', margin='0 10px 0 0'))

        # 3. Right Column: Execution & Results
        
        # Card D: Actions & Logs
        card_exec = VBox([
            HTML("<div class='card-header'>Control Center</div>"),
            w['generate_btn'],
            HBox([w['reset_btn']], layout=Layout(justify_content='flex-end')),
            HTML("<div class='section-label' style='margin-top:15px'>Process Log</div>"),
            VBox([w['log_output']], layout=Layout(classes='console-box'))
        ], layout=Layout(classes='card'))

        # Card E: Visual Preview
        card_preview = VBox([
            HTML("<div class='card-header'>Mesh Preview</div>"),
            w['plot_output']
        ], layout=Layout(classes='card', min_height='300px'))

        right_col = VBox([card_exec, card_preview], layout=Layout(width='60%', margin='0 0 0 10px'))

        # 4. Assembly
        main_body = HBox([left_col, right_col], layout=Layout(align_items='flex-start'))
        
        self.ui = VBox([
            header,
            main_body
        ], layout=Layout(classes='app-container', width='100%', max_width='1100px'))

    def _bind_events(self):
        self.widgets['mode'].observe(self._on_mode_change, names='value')
        self.widgets['generate_btn'].on_click(self.on_generate)
        self.widgets['reset_btn'].on_click(self.on_reset)

    # --- Logic ---

    def _on_mode_change(self, change):
        self._update_visibility()

    def _update_visibility(self):
        mode = self.widgets['mode'].value
        if mode == 'div':
            self.widgets['mesh_size'].layout.display = 'none'
            self.widgets['num_div'].layout.display = ''
        else:
            self.widgets['mesh_size'].layout.display = ''
            self.widgets['num_div'].layout.display = 'none'

    def _sanitize_filename(self, name: str) -> str:
        name = (name or "").strip()
        if not name: return self.default_mesh_basename
        name = re.sub(r"[^\w\-.]+", "_", name)
        return name[:64]

    def on_reset(self, _):
        w = self.widgets
        w['length'].value = 0.005
        w['mode'].value = 'div'
        w['mesh_size'].value = 0.0003
        w['num_div'].value = 20
        w['filename'].value = self.default_mesh_basename
        w['overwrite'].value = False
        with w['log_output']: clear_output()
        with w['plot_output']: clear_output()

    def on_generate(self, _):
        w = self.widgets
        out_log = w['log_output']
        out_plot = w['plot_output']
        
        with out_log:
            clear_output()
            def log(s, color=None):
                style = f"style='color:{color}'" if color else ""
                print(s) # Fallback
                display(HTML(f"<div {style}>{s}</div>"))

            # Validation
            L = w['length'].value
            if L <= 0:
                log("❌ Error: Length must be positive.", "#e74c3c")
                return
            
            name = self._sanitize_filename(w['filename'].value)
            target_path = os.path.join(self.output_dir, f"{name}.msh")
            
            if os.path.exists(target_path) and not w['overwrite'].value:
                log(f"⚠️ File exists: {name}.msh", "#f39c12")
                log("   Check 'Allow Overwrite' to proceed.", "#f39c12")
                return

            # Execution
            log("⚙️ Initializing Generator...", "#3498db")
            os.makedirs(self.output_dir, exist_ok=True)
            
            mode_str = "Use Divisions" if w['mode'].value == 'div' else "Use Mesh Size"
            
            try:
                log(f"• Geometry: L={L}m")
                log(f"• Strategy: {mode_str}")
                
                # Mock calling the backend
                gen = OneDSampleMeshGenerator(
                    length=L,
                    mode=mode_str,
                    mesh_size=w['mesh_size'].value,
                    num_divisions=w['num_div'].value,
                    output_dir=self.output_dir
                )
                
                gen.generate_mesh()
                
                # Handling file rename if needed (backend usually saves as sample_1D.msh default)
                default_file = os.path.join(self.output_dir, f"{self.default_mesh_basename}.msh")
                
                # Logic to rename if user chose a different name
                if name != self.default_mesh_basename:
                    if os.path.exists(default_file):
                        if os.path.exists(target_path): os.remove(target_path)
                        os.rename(default_file, target_path)
                    else:
                        # Maybe the generator already used the target name?
                        pass

                log(f"✅ Success! Saved to:", "#3498db")
                log(f"   {target_path}")
                
                # Plot
                with out_plot:
                    clear_output(wait=True)
                    gen.plot_mesh()
                    
            except Exception as e:
                log(f"❌ Error: {str(e)}", "#c0392b")
                traceback.print_exc()

    def show(self):
        display(self.ui)

if __name__ == "__main__":
    # Example Usage
    app = MeshGUI(output_dir="./gmsh")
    app.show()