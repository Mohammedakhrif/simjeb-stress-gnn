# Stress Field Prediction on Jet Engine Brackets with a Graph Neural Network

This is a small project where I trained a graph neural network (GNN) to predict the stress
field on a jet engine bracket directly from its shape. Normally you get this field from a
finite element (FEA) solve that takes a few minutes. The trained network produces a similar
field in a few milliseconds. I built it to explore how machine learning can speed up
simulation and CAE work.

![FEA vs GNN stress field](stress_field_compare.png)

## What it does

You give the model the surface shape of a bracket and it predicts the von Mises stress at
every surface point, for four standard load cases: vertical, horizontal, diagonal and
torsional. I sample the surface as a cloud of points, connect each point to its nearest
neighbours to form a graph, and run an EdgeConv (DGCNN style) network over it. The network
learns to map local geometry to stress.

## Data

I used SimJEB, a public dataset of 381 jet engine bracket designs. Every bracket comes with an
FEA stress solution for the same four load cases. The load, the supports and the material
(titanium) are identical for all of them, so the only thing that changes is the geometry. That
makes it a clean problem: learn stress from shape. I kept the official train and test split
that ships with the dataset.

> Whalen, E., Beyene, A., Mueller, C. (2021). SimJEB: Simulated Jet Engine Bracket Dataset.
> Harvard Dataverse. CC0. https://simjeb.github.io/

## Method

Each bracket is represented by 8,000 surface points. For every point I use nine features: its
position, an estimated surface normal computed from a local PCA, and three global size numbers
(the log of volume, mass and surface area). Each point is linked to its twelve nearest
neighbours to build the graph. The targets are the von Mises stress for the four loads, which I
transform with log1p and standardize before training. The model has four stacked EdgeConv
layers with a hidden size of 128, followed by a small MLP head, for about 0.23 million
parameters. I trained it with plain MSE loss using Adam at a learning rate of 1e-3 with cosine
annealing, for 250 epochs.

## Results

The model learns the geometry to stress mapping well. The clearest way to see this is the
spatial correlation, which measures whether the predicted field places the high and low stress
regions in the same locations as the real FEA. Across all four load cases it stays high, on the
training brackets and on the held out test brackets the model never saw.

Spatial correlation (median per bracket, predicted vs true field):

| Load case  | Train corr | Test corr |
|------------|:----------:|:---------:|
| Vertical   |    0.81    |   0.64    |
| Horizontal |    0.79    |   0.59    |
| Diagonal   |    0.77    |   0.57    |
| Torsional  |    0.84    |   0.82    |

The torsional case is the standout. On brackets it had never seen, the network reproduces the
torsional stress field with 0.82 correlation, almost matching its training value of 0.84. For
comparison, a baseline that predicts peak stress from global numbers like volume and mass
scores only about 0.04 R2 on torsion, because torsion depends on local shape that those numbers
cannot capture. This is exactly where the graph network earns its place: it reads the geometry
and finds where the stress concentrates.

For completeness, here are the absolute accuracy metrics on the test set, where the model also
reports R2 and mean absolute error in MPa:

| Load case  | Test R2 | Test MAE (MPa) |
|------------|:------:|:--------------:|
| Vertical   |  0.17  |      67        |
| Horizontal |  0.30  |      63        |
| Diagonal   |  0.25  |      48        |
| Torsional  |  0.43  |      28        |
| Overall    |  0.29  |      n/a       |

Getting the exact stress magnitude is harder than getting the spatial pattern, especially with
only about 300 brackets for training, so there is some overfitting (overall R2 is around 0.69
on the training set against 0.29 on the test set). I am keeping that visible on purpose. For a
tool meant to screen many designs quickly and flag the likely problem areas, finding where the
stress concentrates is the property that matters most, and the strong correlation shows the
model does that reliably. It also points to the obvious next improvement, which is more data.

## What I would try next

Train on more shapes, such as the larger DeepJEB dataset, and add regularization and early
stopping to close the gap between training and test. Replace the nearest neighbour links with
the real mesh connectivity. Feed the load and support conditions in as inputs so the model can
generalize beyond the four fixed cases. And eventually place the surrogate inside a shape
optimization loop, where a fast stress estimate is most valuable.

## How to run

Open `simjeb-stress-gnn.ipynb` in Google Colab on a GPU runtime. Download the SimJEB
`dataverse_files.zip` to your Drive and point the `ZIP` path in the setup cell at it, then run
the cells from top to bottom. The data extraction is cached after the first run.

```
pip install -r requirements.txt
```

I built this as a portfolio project around something I care about, which is bringing machine
learning into simulation and CAE work to make engineering faster.
