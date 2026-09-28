# OptiCam

**Camera-placement optimization using spatial visibility modelling and mixed-integer programming.**

## Project overview

OptiCam explores where to place cameras, and which direction to face them, to maximize indoor spatial coverage under a limited camera budget. It turns annotated indoor point clouds into occupancy grids, computes obstacle-aware visibility, and uses **CVXPY** to express camera selection as a binary optimization problem.

The project demonstrates an optimization-based AI engineering workflow: translate geometry into a mathematical representation, encode operational constraints, solve independent budget scenarios in parallel, and visualize the trade-off between equipment count and covered space. The implementation uses geometric modelling and mathematical optimization rather than a trained prediction model.

**Repository scope:** five research notebooks and a standalone solver script. Notebook outputs preserve exploratory results, but the source dataset, generated visibility matrices, selected-room data, and serialized solver results are not included. Reproducing the complete experiment requires those inputs and an assembly step described below.

## Problem formulation

The solver implements a **binary maximum-coverage problem**, a special case of mixed-integer linear optimization.

Let:

- $I$ be the set of candidate camera configurations. A directional candidate identifies a room, grid location, and orientation.
- $J$ be the set of target free-space cells.
- $V_{ij} \in \{0,1\}$ indicate whether candidate $i$ can see cell $j$.
- $K$ be the maximum number of cameras.
- $I_r$ contain the candidates belonging to room $r$.

The binary decision variables are $x_i$, indicating whether a camera configuration is selected, and $y_j$, indicating whether a target cell is counted as covered.

$$
\begin{aligned}
\max_{x,y}\quad & \sum_{j \in J} y_j \\
\text{subject to}\quad
& \sum_{i \in I} x_i \le K, \\
& \sum_{i \in I_r} x_i \ge 1 && \text{for every represented room } r, \\
& y_j \le \sum_{i \in I} V_{ij}x_i && \text{for every target cell } j, \\
& x_i,y_j \in \{0,1\}.
\end{aligned}
$$

The objective counts each covered cell once, even when several cameras see it. Cells with no visible candidate are explicitly constrained to zero. Maximization makes a visible cell count as covered without requiring a separate lower-bound constraint.

[`solver.py`](solver.py) applies the per-room placement requirement to rooms present in `camera_positions`. A budget smaller than that room count is infeasible. The model does not require complete coverage of each room, and it does not prevent selecting multiple orientations at the same grid location; each selected orientation consumes one unit of budget.

## Approach

### 1. Convert point clouds to a spatial representation

[`preprocessing.ipynb`](preprocessing.ipynb) reads annotation files from `Stanford3dDataset_v1.2_Aligned_Version`, iterating over `Area_1` through `Area_6`. It projects non-floor, non-ceiling object points onto the XY plane and rasterizes them at **0.2 metres per cell**. Occupied cells are `1`; unoccupied cells are `0`.

It saves per-room NumPy grids, PNG previews, and a CSV inventory under `Processed_Area/`. Grid bounds come from obstacle coordinates; zero-valued cells are not independently checked against a floor mask.

### 2. Model visibility and occlusion

Two notebooks explore complementary visibility models:

| Model | Candidate | Visibility rule |
| --- | --- | --- |
| Basic | Every free grid cell | Target is within 25 cells (5 m), with an unobstructed rasterized line of sight. |
| Directional | Every free grid cell at each of `0°`, `90°`, `180°`, `270°` | The basic range/occlusion checks plus a 90° viewing cone. |

Both use `skimage.draw.line` to trace the cells between source and target. Any occupied cell blocks visibility. Directional angles use grid row/column coordinates, so plots should be interpreted in that coordinate convention.

Per-room visibility maps are stored in compressed `.npz` files. Basic maps use `(row, column)` keys; directional maps use `(row, column, angle)` keys. The standalone solver consumes a global binary visibility matrix built from these maps, but the matrix-assembly code is not included.

### 3. Optimize and compare camera budgets

[`solver.py`](solver.py) creates Boolean CVXPY variables and routes the model to **Gurobi**, **CBC**, or **GLPK_MI**. Its defaults evaluate budgets **21 through 40** using Gurobi.

A `ProcessPoolExecutor` runs independent budget problems with `min(cpu_count(), 4)` workers. Results are collected with `as_completed`, tracked with `tqdm`, sorted by budget, and serialized with `pickle`. This parallelizes the budget sweep; it does not partition a single optimization problem across workers. Despite the script's “Reducing problem size” message and output filename, the current code passes through the full input arrays unchanged.

### 4. Analyze coverage and placement

[`Result_Plotting.ipynb`](Result_Plotting.ipynb) plots coverage against camera budget, calculates marginal coverage gains, and flags the first gain below **one percentage point**. A second cell visualizes selected camera locations and viewing cones, either for a chosen budget (`40` by default) or a budget suggested by `KneeLocator`.

## Tech stack

| Technology | Purpose |
| --- | --- |
| Python and Jupyter | Solver implementation and exploratory workflow |
| NumPy | Point-cloud arrays, occupancy grids, visibility data, and `.npy`/`.npz` storage |
| scikit-image | Rasterized line-of-sight checks |
| CVXPY | Boolean variables, linear constraints, and objective modelling |
| Gurobi / CBC / GLPK_MI | Explicitly supported mixed-integer solver backends |
| `concurrent.futures`, `multiprocessing` | Process-based evaluation of independent camera budgets |
| Matplotlib | Point-cloud projections, coverage curves, grids, and camera overlays |
| pandas, tabulate | Preprocessing inventory and coverage tables |
| kneed | Optional elbow selection from the coverage curve |
| tqdm, pickle | Progress reporting and serialization of solver results |

## Repository structure and notebook guide

```text
.
├── README.md
├── solver.py
├── initial_look.ipynb
├── preprocessing.ipynb
├── basic_visibility.ipynb
├── directional_visibility.ipynb
└── Result_Plotting.ipynb
```

| File | Purpose and outputs |
| --- | --- |
| [`initial_look.ipynb`](initial_look.ipynb) | Single-room prototype using `Area_1/conferenceRoom_1`: visualizes floor, walls, and chairs; builds a 0.1 m grid from walls/chairs; computes omnidirectional visibility; solves a `K=10` model with GLPK_MI; plots coverage. It does not include the standalone solver's per-room constraints. |
| [`preprocessing.ipynb`](preprocessing.ipynb) | Batch point-cloud-to-grid conversion at 0.2 m resolution across six areas; writes grid arrays, images, and `occupancy_grid_summary.csv`. |
| [`basic_visibility.ipynb`](basic_visibility.ipynb) | Generates omnidirectional visibility maps for all processed room grids and visualizes a sample source in `Area_3/lounge_2`. |
| [`directional_visibility.ipynb`](directional_visibility.ipynb) | Generates maps for four orientations and a 90° field of view; illustrates the difference between the ideal viewing cone and cells actually visible around obstacles. |
| [`solver.py`](solver.py) | Loads the global visibility matrix, solves each camera budget, and writes results and input metadata to a pickle file. |
| [`Result_Plotting.ipynb`](Result_Plotting.ipynb) | Reads merged solver results; produces coverage/marginal-gain analysis and per-room placement overlays. Requires additional result metadata and selected-room files. |

## How to run the solver

### 1. Set up Python and a mixed-integer backend

From a Python environment compatible with your chosen CVXPY release:

```bash
git clone https://github.com/niki-21/opticam.git
cd opticam
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install cvxpy numpy tqdm
```

Install one of the supported backends separately:

| `solver` value | Backend setup |
| --- | --- |
| `"GUROBI"` | Install `gurobipy` and configure a Gurobi license appropriate for the problem size and execution environment. This is the script default. |
| `"CBC"` | Install `cylp` and its platform-specific CBC prerequisites. |
| `"GLPK_MI"` | Install `cvxopt` with GLPK support; GLPK is bundled with CVXOPT on many platforms. |

Follow the [CVXPY solver installation guide](https://www.cvxpy.org/install/index.html) for platform-specific instructions. Verify discovery before launching the sweep:

```bash
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

The chosen backend must appear in that list. The script does not automatically fall back to another solver. Dependencies are not pinned in this repository.

For the notebooks, also install:

```bash
python -m pip install jupyterlab matplotlib pandas scikit-image tabulate kneed
jupyter lab
```

### 2. Prepare the input archive

Run the preprocessing notebook and the visibility notebook matching your model after obtaining the expected annotated dataset. Basic and directional visibility generation are alternative branches; the directional notebook does not require basic visibility output.

The standalone solver expects **`global_selected_visibility_final.npz` in the current working directory**, with these keys:

| Key | Contract |
| --- | --- |
| `V` | A binary array of shape `(number_of_candidates, number_of_target_cells)`. A value of `1` means that candidate sees that target. |
| `camera_positions` | One entry per matrix row, unpackable as `(room_id, camera_configuration)`. Directional plotting expects the configuration to be `(row, column, angle)`. |
| `cells` | One entry per matrix column, in exactly the same order as the target-cell indexing used to build `V`. The solver uses its length. |

**The archive and its assembly script are missing from the repository.** To supply it, choose the rooms, assign unique room identifiers across areas, enumerate camera configurations and target cells in a stable order, then convert each room's visibility dictionary into the corresponding matrix entries. Preserve that ordering in both metadata arrays. Do not infer global column indices from per-room coordinates alone.

### 3. Configure and run

Edit the constants in the `__main__` block of `solver.py` as needed:

```python
Ks = list(range(21, 41))
max_workers = min(cpu_count(), 4)
solver = "GUROBI"  # Also accepts "CBC" or "GLPK_MI"
```

Choose feasible budgets and a worker count appropriate for memory and solver-license limits. Then run from the directory containing the input archive:

```bash
python solver.py
```

With the default backend, the output is `GUROBI_reduced_async_Final2.pkl`. The filename changes with the backend name. Its dictionary contains:

- `results`: sorted `(K, covered_cell_count, x_values)` tuples.
- `num_cells`: the total number of target cells.
- `npz_file`: the input archive filename.

Coverage percentage is `100 * covered_cell_count / num_cells`. Exceptions raised during `prob.solve` return `(K, 0, None)`; these are failed solves, not evidence of zero achievable coverage. The code does not validate solver status before converting the objective to an integer, so infeasible or unbounded outcomes can still terminate the sweep.

### 4. Connect results to the plotting notebook

`Result_Plotting.ipynb` currently loads `GUROBI_merged.pkl`, not the filename written by `solver.py`. For a new run, point its loading cells to your output file. The coverage-curve cell uses `results` and `num_cells` directly.

The overlay cell additionally expects `cam_map`, which `solver.py` does not save. For an unchanged full matrix, the index mapping is `np.arange(len(camera_positions))`; if candidates were reduced or reordered, supply the actual mapping. It also needs the corresponding grids and conical-visibility files under `Selected_Rooms/`. No result-merging or selected-room preparation script is included.

The existing loaders use pickle-enabled NumPy archives, Python pickle, and `eval` for visibility-map keys. Run them only with trusted artifacts.

## Results retained in the notebooks

The saved output in [`Result_Plotting.ipynb`](Result_Plotting.ipynb) reports the following points from a Gurobi-labelled budget sweep:

| Camera budget | Cells covered | Coverage |
| --- | --- | --- |
| 5 | 932 | 26.78% |
| 20 | 2,438 | 70.06% |
| 25 | 2,643 | 75.95% |
| 40 | 3,043 | 87.44% |

The notebook identifies `K=25` as the first point where the marginal improvement falls below one percentage point: **0.98 percentage points** from `K=24`. This is its chosen saturation heuristic, not proof that additional cameras are unnecessary or that 25 is a uniquely optimal budget.

These values are preserved notebook output, not a fresh benchmark. The underlying `GUROBI_merged.pkl`, visibility archive, room selection, and solve-status logs are absent. The saved sweep spans budgets 5–40, whereas the current solver defaults to 21–40. No runtime, parallel speedup, or cross-backend performance comparison is established by the repository.

## Engineering concepts demonstrated

- **Problem-to-model translation:** express a spatial planning task through candidate sets, a visibility matrix, binary variables, and operational constraints.
- **Geometric preprocessing:** project point clouds, discretize space, model occlusion, and add directional field-of-view constraints.
- **Optimization-based decision making:** solve a maximum-coverage integer program through a common CVXPY interface to multiple backends.
- **Parallel experiment execution:** evaluate independent budgets in worker processes and restore ordered results after asynchronous completion.
- **Trade-off analysis:** quantify marginal coverage gains and inspect placements visually before interpreting a budget choice.
- **Artifact interfaces:** pass geometry and optimization results through NumPy archives and serialized metadata, with explicit indexing requirements.

## Limitations and next steps

The model is a two-dimensional approximation: it ignores camera height, vertical field of view, image quality, mounting feasibility, and dynamic obstacles. Visibility is discretized to a fixed range and four orientations. Pairwise visibility generation examines free-cell pairs, which can become expensive for large rooms.

Reproducibility requires supplying missing data and assembly steps, reconciling the plotting interfaces, and recording package versions and solver status. The repository has no automated tests, CI configuration, dependency lockfile, or solver time/gap controls. Improvements such as floor masking, distinct-location constraints, sparse visibility construction, and robust infeasibility handling are future work rather than implemented features.
