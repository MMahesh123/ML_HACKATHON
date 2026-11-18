# PathSnap Dataset Directory

This directory stores the dataset partitions for the PathSnap challenge:

```text
data/
├── raw/
│   ├── train/            # 80 training trips (.track, .nodes, .arcs, .route)
│   ├── student_test/     # 10 student-testing trips (.track, .nodes, .arcs, .route)
│   └── evaluator_test/   # 10 evaluator-testing trips (.track, .nodes, .arcs)
└── processed/            # Cached candidate sets and extracted feature matrices
```

## Raw File Format Reference

1. **`.track`**:
   - 3 whitespace-separated columns without headers: `longitude`, `latitude`, `timestamp` (in seconds).
   - Nominally 1 Hz (~97% of steps are exactly 1 second apart).
2. **`.nodes`**:
   - 2 whitespace-separated columns without headers: `longitude`, `latitude`.
   - 0-based row index represents `node_id`.
3. **`.arcs`**:
   - 2 integer columns without headers: `from_node_id`, `to_node_id`.
   - 0-based row index represents `arc_id`.
4. **`.route`**:
   - 1 integer column without headers: chronological sequence of ground-truth `arc_id` values.

## Data Isolation Policy
- Observations from the same trip must **never** be distributed across different partitions.
- Evaluator testing data must **never** be used for model training, feature selection, or hyperparameter tuning.
- Ground-truth `.route` files for evaluator trips remain strictly server-side on Laptop 1.
