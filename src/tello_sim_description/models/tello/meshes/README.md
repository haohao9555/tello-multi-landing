# Detailed Tello-inspired visuals

Original procedural geometry authored for this project, in metres. This is an illustrative Tello-inspired model, not official DJI CAD or a dimensionally certified digital twin. No third-party drone mesh was imported.

Run `python tools/build_tello_visuals.py` from the project root to regenerate the OBJ meshes and visual elements. NumPy and Matplotlib are required for regeneration; Gazebo only needs the generated OBJ files at runtime. The top-shell lettering uses the installed DejaVu Sans outline (see `LICENSE-DejaVu.txt`).

Features include a smooth white shell, dark lower chassis, battery panel and label, camera bezel/lens, vents, LED detail, motor housings and collars, sculpted two-blade propellers, guard rings and braces, screws, and connected landing supports. The fleet colour is applied to the identification panel rather than the whole shell.

All changes to `model.sdf` are restricted to `visual` elements. Links, link poses, mass/inertial matrices, collision geometry, joints, motor constants, control gains and topics retain their original values. Visual geometry extends beyond the original box collider; detailed guard collision and aerodynamic effects are intentionally not inferred from the mesh.

The browser console offers `Fleet overview` and `Aircraft close-up` camera views, with an independent camera target selection. Tello defaults to Gazebo rendering. M30 remains on Fast simulation because its Gazebo dynamics/model are not implemented in this project.
