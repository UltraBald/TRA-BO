# TRA-BO

TRA-BO is Bayesian optimization with Temporal Rank Agreement for time-series
forecasting. It estimates whether the performance ranking of evaluated
configurations persists from earlier to recent validation windows, then uses
that estimate to calibrate the Gaussian-process target and guide acquisition-
function selection.

## Method

For configuration $\lambda_i$, let $\ell_{ik}$ be its loss on validation
window $V_k$. With $K-m$ earlier windows and $m$ recent windows,

$$
e_i = \frac{1}{K-m}\sum_{k=1}^{K-m}\ell_{ik}, \qquad
r_i = \frac{1}{m}\sum_{k=K-m+1}^{K}\ell_{ik}, \qquad
g_i = \frac{1}{K}\sum_{k=1}^{K}\ell_{ik}.
$$

The Temporal Rank Agreement is

$$
q = \frac{1 + \tau_b\!\left((e_i)_{i=1}^{n}, (r_i)_{i=1}^{n}\right)}{2},
$$

where $\tau_b$ is Kendall's rank correlation coefficient.

The GP is fitted to the calibrated target

$$
\omega(q) = \eta(1-q), \qquad
y_i(q) = [1-\omega(q)]g_i + \omega(q)r_i.
$$

The default settings are `eta=1/8`, `n_initial=16`, `recent_count=2`, and
`block_size=5`. The GP is refitted after every trial. TRA is recomputed after
each block of five trials and at the final budget; each update recalibrates the
targets of all evaluated configurations.

At every BO trial, the LLM controller receives the current TRA, BO diagnostics,
and acquisition history. It selects PI, EI, UCB, or Thompson sampling. The
selected acquisition function scores the remaining candidate pool and chooses
the next configuration. The LLM never chooses a configuration directly.

## Installation

```bash
python3 -m pip install -e .
```

Python 3.9 or later is required.

## LLM configuration

The controller uses an OpenAI-compatible `/chat/completions` endpoint:

```bash
export TRABO_API_KEY="your-api-key"
export TRABO_BASE_URL="https://your-provider.example/v1"
export TRABO_MODEL="your-model-id"
```

The equivalent `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`
variables are also accepted. Optional settings are `TRABO_TEMPERATURE`,
`TRABO_MAX_TOKENS`, `TRABO_TIMEOUT`, and `TRABO_MAX_RETRIES`.

## Python API

The objective must return one loss for each ordered, visible validation window.
It must not return the held-out future-window loss.

```python
import numpy as np

from tra_bo import LLMAcquisitionSelector, TRABO, TRABOConfig


candidate_pool = np.load("candidate_pool.npy")


def evaluate_windows(candidate: np.ndarray) -> list[float]:
    # Train on the prefix preceding each validation window.
    return [loss_v1, loss_v2, loss_v3, loss_v4]


optimizer = TRABO(
    TRABOConfig(
        budget=50,
        n_initial=16,
        recent_count=2,
        block_size=5,
        eta=1 / 8,
        random_state=17,
    ),
    selector=LLMAcquisitionSelector(),
)
result = optimizer.optimize(candidate_pool, evaluate_windows)
result.save_json("run.json")
```

Use `initial_indices=` to share an initialization across optimizers.
After optimization, evaluate future performance outside the optimizer:

```python
from tra_bo import incumbent_future_loss_trajectory, normalized_future_regret_auc

future_trajectory = incumbent_future_loss_trajectory(
    result.incumbent_indices,
    candidate_future_losses,
)
rauc = normalized_future_regret_auc(future_trajectory, candidate_future_losses)
```

## Precomputed validation losses

When each candidate has already been evaluated on the visible windows, run
TRA-BO directly from two aligned matrices:

```bash
tra-bo optimize \
  --candidates candidate_pool.npy \
  --validation-losses validation_losses.npy \
  --output run.json \
  --selector llm \
  --seed 17
```

`candidate_pool.npy` has shape `[n_candidates, dimension]` and
`validation_losses.npy` has shape `[n_candidates, K]`. Supported input formats
are `.npy`, `.json`, `.csv`, and whitespace-delimited `.txt`.

For a fixed-acquisition run, set `--selector EI`, `PI`, `UCB`, or `TS`.

## Search spaces and temporal splits

`generate_sobol_pool()` creates the latent candidate matrix and decoded DLinear
or PatchTST configurations. `make_ordered_split()` creates four chronological
validation windows followed by one held-out future window. The configuration
files in `configs/` contain the optimizer defaults, benchmark tasks, datasets,
search spaces, and training settings.

## Project structure

```text
src/tra_bo/
├── temporal.py      # TRA, early/recent/full-history losses, calibrated targets
├── surrogate.py     # ARD Matérn-5/2 Gaussian process
├── acquisition.py   # PI, EI, UCB, and Thompson sampling
├── llm.py           # LLM controller and API client
├── optimizer.py     # blockwise TRA-BO loop
├── spaces.py        # DLinear and PatchTST Sobol spaces
├── splits.py        # chronological validation and future windows
├── evaluation.py    # normalized future-regret metrics
└── cli.py           # matrix-based command-line entry point
configs/             # defaults and benchmark configuration
docs/                # LLM controller interface
schemas/             # controller and result JSON schemas
tests/               # algorithm and interface tests
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

MIT
