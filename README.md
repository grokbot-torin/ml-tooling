# tensor-toolkit

A small, dependency-light toolkit for everyday array work in Python. It focuses on NumPy-native helpers for tensor arithmetic, data-pipeline transforms, feature encoding, tokenization, and sampling — the kind of utilities you usually end up rewriting in notebooks.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and NumPy.

## Quick start

```python
import numpy as np
from tensor_toolkit import batch_l2_normalize, sliding_window

x = np.random.randn(32, 128)
normalized = batch_l2_normalize(x)
windows = sliding_window(normalized, window=8, stride=4)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

CI runs the same suite on Python 3.10 and 3.12 for every push and pull request to `main`.

## Layout

- `tensor_toolkit.ops` — array / tensor helpers
- `tensor_toolkit.pipeline` — composable transforms
- `tensor_toolkit.augment` — simple numeric augmentation
- `tensor_toolkit.encode` — feature encoding
- `tensor_toolkit.tokenize` — lightweight tokenization helpers
- `tensor_toolkit.sample` — sampling utilities

## License

MIT
