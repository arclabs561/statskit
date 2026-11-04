# statskit examples

Examples demonstrate API usage on small hand-written inputs. Their printed
numbers are not benchmark evidence or sklearn parity checks; those live in
`tests/` (`tests/rosetta_*.rs` compare against sklearn reference values).

## Where to start

| I want to... | Run |
|---|---|
| Evaluate a binary classifier end to end: hard-label metrics, probabilistic metrics, calibration, and a McNemar comparison of two systems | `classification_eval` |

```sh
cargo run --example classification_eval
```

## All examples

| Example | What it shows |
|---|---|
| `classification_eval` | Accuracy, precision/recall/F1, MCC, kappa, log loss, ROC AUC, average precision, Brier score, ECE/MCE, and McNemar's test on one 10-sample binary problem |

No example requires a cargo feature.
