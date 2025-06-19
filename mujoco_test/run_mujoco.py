"""
Usage:
1. Save this script as `run_mujoco.py` in the root of your project.
2. Ensure your scene XML is at `data/go2/scene.xml` or `data/go2/go2.xml`.
3. To run with a visible window:
      python run_mujoco.py
4. To run headless (no window):
      export HEADLESS=true
      python run_mujoco.py

This script will automatically:
- Remove unsupported `autolimits` attributes from the XML by regex.
- Save cleaned XML next to original, preserving include paths.
- Choose rendering backend via MUJOCO_GL (default 'egl') or HEADLESS.
- Launch MuJoCo simulation for 1000 steps, rendering each frame.
"""
import os
import re

# 0. Set rendering backend (egl or osmesa)
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')

# 1. Locate original XML
src_paths = [
    "data/go2/scene.xml",
    "data/go2/go2.xml"
]
for p in src_paths:
    if os.path.exists(p):
        src_path = p
        break
else:
    raise FileNotFoundError(f"Cannot find scene XML in {src_paths}")

# 2. Read original XML as text and remove the entire autolimits attribute or invalid compiler tag
with open(src_path, 'r') as f:
    xml_text = f.read()
# Remove autolimits attributes within compiler tags
dirty = re.sub(r'<compiler[^>]*>', '<compiler>', xml_text)
# Also remove any leftover autolimits occurrences
dirty = re.sub(r"autolimits=\"[^\"]*\"", '', dirty)
cleaned_text = dirty

# 3. Write cleaned XML back to original directory so includes resolve correctly Write cleaned XML back to original directory so includes resolve correctly
dirpath = os.path.dirname(src_path)
cleaned_name = os.path.join(dirpath, "scene_cleaned.xml")
with open(cleaned_name, 'w') as f:
    f.write(cleaned_text)

# 4. Compute absolute path to cleaned XML
abs_cleaned = os.path.abspath(cleaned_name)

# 5. Import MuJoCo and create simulation
import mujoco_py
from mujoco_py import load_model_from_path, MjSim

model = load_model_from_path(abs_cleaned)
sim = MjSim(model)

# 6. Choose viewer based on HEADLESS env var
headless = os.environ.get('HEADLESS', 'false').lower() in ('1','true','yes')
if headless:
    from mujoco_py import MjRenderContextOffscreen as Viewer
    viewer = Viewer(sim, 0)
else:
    from mujoco_py import MjViewer as Viewer
    viewer = Viewer(sim)

# 7. Run simulation for 1000 steps and render
for _ in range(1000):
    sim.step()
    viewer.render()
