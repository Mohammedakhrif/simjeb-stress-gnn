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

The table below is measured on the 77 test brackets the model never saw during training. The
corr column is the median per bracket correlation between the predicted and the true field. In
plain terms, it tells you whether the model puts the high stress regions in the right place.

| Load case  | Test R2 | Test MAE (MPa) | Test spatial corr |
|------------|:------:|:--------------:|:-----------------:|
| Vertical   |  0.17  |      67        |       0.64        |
| Horizontal |  0.30  |      63        |       0.59        |
| Diagonal   |  0.25  |      48        |       0.57        |
| Torsional  |  0.43  |      28        |       0.82        |
| Overall    |  0.29  |      n/a       |       n/a         |

The result I find most useful is on the torsional load. A simple baseline that only looks at
global numbers like volume and mass is almost blind to torsion, scoring around 0.04 R2, because
torsional stress depends on local shape that those numbers cannot capture. The GNN reaches 0.82
correlation on the torsional field for brackets it has never seen. In general the model is
better at finding where stress concentrates than at getting the exact value, which is what you
want from a fast tool that screens designs and flags the likely problem areas.

## Honest limitations

The dataset is small, only about 300 brackets for training, so the model overfits. The overall
R2 is around 0.69 on the training set and 0.29 on the test set, and the spatial correlation
holds up much better than the absolute R2. The graph uses nearest neighbour links because the
dataset does not include the real surface mesh, so the connectivity is approximate. The four
outputs are the four fixed load cases from the dataset, so the model does not yet take an
arbitrary load or material as input.

## What I would try next

Train on more shapes, such as the larger DeepJEB dataset, and add regularization and early
stopping to reduce the overfitting. Replace the nearest neighbour links with the real mesh
connectivity. Feed the load and support conditions in as inputs so the model can generalize
beyond the four fixed cases. And eventually place the surrogate inside a shape optimization
loop, where a fast stress estimate is most valuable.

## How to run

Open `simjeb-stress-gnn.ipynb` in Google Colab on a GPU runtime. Download the SimJEB
`dataverse_files.zip` to your Drive and point the `ZIP` path in the setup cell at it, then run
the cells from top to bottom. The data extraction is cached after the first run.

```
pip install -r requirements.txt
```

I built this as a portfolio project around something I care about, which is bringing machine
learning into simulation and CAE work to make engineering faster.
