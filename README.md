# Dynamic Community Detection with ML-Guided Louvain

Incremental community detection on evolving graphs — instead of re-running Louvain from scratch every time the graph changes, a trained ML model predicts which nodes are likely to shift communities and processes only those.

## The Problem

Standard Louvain is expensive on large graphs. When you're dealing with a graph that changes continuously (social networks, citation graphs, infrastructure networks), running full community detection on every update doesn't scale. The incremental approach keeps the previous partition and only re-evaluates nodes near the change.

## What's in the Notebook

The notebook (`dynamic_community_detection_clean.ipynb`) walks through the full development arc across 8 sections:

| Section | What it covers |
|---|---|
| 1. MLPriorityLouvain | Core idea: RF classifier scores nodes by change probability, drives a max-heap priority queue |
| 2. Validation | Stochastic block model test — NMI and ARI both hit 1.0 on clean synthetic data |
| 3. Quality Comparison | Incremental vs. global Louvain on perturbed graphs (NMI ~0.74) |
| 4. Benchmark | Speed vs. quality table across 5 graphs |
| 5. Pre-trained Model | Train once across multiple graphs for a more general predictor |
| 6. Full Pipeline | Clean class-based refactor: `DynamicGraphGenerator → CommunityChangePredictor → MLFastLouvain → Evaluator` |
| 7. Large Graph Demo | Where incremental actually wins — **~38x speedup** at 2000 nodes with localized changes |
| 8. ML + Improved Fast Louvain | Combines tree-structure splitting (Zhang et al. 2021) with ML priority ordering |

## Approaches Compared

Three incremental strategies are evaluated throughout:

**`MLPriorityLouvain`** — trains a `RandomForestClassifier` on node features (degree, internal ratio, participation coefficient, clustering coefficient) to predict community instability. High-probability nodes go first in the update queue.

**`OptimizedIncrementalLouvain`** — replaces the ML model with a fast heuristic (boundary fraction + degree boost). No training overhead. This is the one that achieves 38x speedup on large graphs.

**`MLEnhancedImprovedLouvain`** — adds tree-structure detection from Zhang et al. (2021) on top. Strips peripheral chains from the graph before clustering, then adds them back. ML priority ordering guides the hierarchical phase.

## Results

| Approach | Graph size | Speedup vs. full Louvain | Modularity retained |
|---|---|---|---|
| `MLPriorityLouvain` | ~200 nodes | ~1–1.5x | ~74% NMI |
| `OptimizedIncrementalLouvain` | 2000 nodes | **~38x** | ~100% |
| `MLEnhancedImprovedLouvain` | 500–2000 nodes | 0.07–0.1x (small graphs) | ~97–100% |

The heuristic predictor punches above its weight. The ML model starts paying off when graphs are large enough that RF inference is cheap relative to Louvain iteration, or when updates are chained over many timesteps and the model amortizes its training cost.

## Dependencies

```
networkx
numpy
scikit-learn
pandas
```

## Usage

Open the notebook and run sections sequentially. Each section is self-contained — you can jump to Section 7 for the large-graph benchmark without running the earlier cells.

To run the full ML pipeline from scratch:

```python
gen = DynamicGraphGenerator()
training_graphs = gen.generate_training_dataset(n_graphs=100, n_nodes=150)

predictor = CommunityChangePredictor()
predictor.train(training_graphs)

# Then test on new graphs
test_graphs = gen.generate_training_dataset(n_graphs=100, n_nodes=150)
results = Evaluator.test_incremental_algorithm(predictor, test_graphs)
Evaluator.print_summary(results)
```

## Notes

- The `DynamicCommunityPredictor` in Sections 1–2 depends on external modules (`Improved_Louvain_Algo`, `PredictionModel`) not included here — those cells show outputs from a previous run
- The best standalone demo is Section 7 (`OptimizedIncrementalLouvain`) — fully self-contained and shows the clearest speedup
- Localized modifications (changes clustered near an epicenter) favor incremental dramatically; random global perturbations narrow the gap
