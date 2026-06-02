# simjeb-stress-gnn
A geometry-aware GNN surrogate for FEA stress prediction, built on the SimJEB bracket dataset.
# Predicting stress on jet-engine brackets with a Graph Neural Network

I work in simulation (CAE), and I wanted to find out how far modern machine learning could go at speeding up the slow part of the job: running structural analyses. This project is where I ended up — a model that looks at the 3D shape of a bracket and predicts where it will be stressed, in milliseconds instead of the minutes a finite-element solve normally takes.

It's a learning project, and I've tried to be honest throughout about what it does well and where it falls short.

![Predicted vs FEA stress field](stress_field_compare.png)

*Left column: the true FEA stress. Right column: my model's prediction. Three held-out test brackets under torsional load — the predicted hot-spots land where the real ones are.*

## The idea

In design work, every time you change a geometry you have to re-run the simulation, so in practice you only ever test a handful of variants. If a model could *approximate* the solver quickly and accurately, you could screen many more designs in the same time. That kind of fast stand-in is called a surrogate model — and the catch is that stress depends on *where* the material sits, not just how much there is. So a useful surrogate has to actually read the shape. A graph neural network can; a model that only sees summary numbers can't. Testing that contrast is the whole point of the project.

## The data

I used [SimJEB](https://simjeb.github.io/) — 381 real bracket designs submitted to the GE Jet Engine Bracket Challenge, each cleaned, meshed, and simulated with FEA under four standardized load cases (vertical, horizontal, diagonal, torsional). Every bracket shares the same mounting interface, loads, and material, so the only thing that varies between samples is the geometry. That's what lets the model focus purely on shape.

## What I built

Two models, on purpose, so I had something honest to compare against:

**A classical baseline.** A random forest that predicts the maximum stress per load case from global descriptors only — volume, mass, surface area, and so on. This is the geometry-*blind* reference.

**A graph neural network.** I represented each bracket's surface as a graph: ~8,000 surface points as nodes, each connected to its nearest neighbours, with node features of position, surface normal, and overall size. An EdgeConv (DGCNN-style) message-passing network then predicts the von Mises stress at every node, for all four load cases at once. I trained it with a plain MSE loss on log-scaled, standardized stress, using Adam with a cosine learning-rate schedule.

I used the dataset's own train/test split (304 train / 77 test) so the numbers are reproducible.

## Results (held-out test set)

| Load case  | Baseline R² | GNN R² | GNN spatial correlation\* |
|------------|-------------|--------|---------------------------|
| Vertical   | 0.33        | 0.24   | 0.66 |
| Horizontal | 0.46        | 0.35   | 0.60 |
| Diagonal   | 0.26        | 0.26   | 0.58 |
| Torsional  | **0.04**    | 0.43   | **0.82** |
| Overall    | 0.27        | 0.32   | — |

*\*Median per-bracket correlation between predicted and true node stress — how well the model reproduces the spatial pattern of stress.*

The single result I find most telling is **torsional**. The classical baseline is almost completely blind to it (R² = 0.04), because torsional stress is governed by how the material wraps around the load path — pure geometry that summary numbers throw away. The GNN, which actually reads the shape, is strongest exactly there, reproducing the stress pattern with 0.82 correlation on brackets it never saw during training. That inversion — best where the simple model is worst — is the core of the story.

## Being honest about the limits

I checked train vs test, and the model overfits: overall R² is 0.66 on training brackets but 0.32 on test brackets. With only 304 training shapes and a few hundred thousand parameters, the network partly memorizes what it's seen. That's expected, and worth stating plainly.

The encouraging counterpart: the *spatial* correlation barely drops between train and test (torsional: 0.835 to 0.821). So while exact stress *magnitudes* are hard to generalize, the model's sense of *where* stress concentrates holds up well on unseen geometry. In other words, it's better as a "show me the danger zones" tool than as a precise magnitude predictor — which, for early-stage design screening, is often exactly what you want.

## What I'd try next

- Train on **DeepJEB** (a 2,000+ bracket augmented version of this dataset) to lift the data ceiling.
- Add a validation set with dropout, weight decay, and early stopping to close the overfitting gap.
- Use the true FE mesh connectivity (instead of a nearest-neighbour graph) and a full MeshGraphNets architecture.
- Encode the load and support locations as node features so the model can handle loads it wasn't trained on.
- Wrap the trained surrogate in an optimization loop to search for lighter, stronger brackets.

## Running it

The full pipeline is in the notebook. To set up:

```
pip install -r requirements.txt
```

Then open the notebook and run the cells in order (a GPU runtime is recommended for training).

## Credit

SimJEB is released under CC0 / the Open Data Commons Attribution License. All credit for the dataset goes to its authors:

> E. Whalen, A. Beyene, and C. Mueller, "SimJEB: Simulated Jet Engine Bracket Dataset," *Computer Graphics Forum* (Eurographics Symposium on Geometry Processing), 2021. arXiv:2105.03534.
