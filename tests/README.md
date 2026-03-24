#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 22/03/26

# Tests — Interactive Neural Network Dashboard

## Are these test bodies pytest tests?

**Yes.** Every test in this directory is a **pytest** test.  The key indicators are:

- `conftest.py` — pytest's special fixture-sharing file; fixtures defined here are injected
  automatically into any test function that names them as a parameter.
- Test functions are discovered by pytest because their names begin with `test_`.
- Test classes are discovered because their names begin with `Test`.
- `pytest.skip("Not yet implemented")` — a pytest-specific call that marks a test as
  *pending* rather than passing or failing.

Running the suite with `python -m pytest tests/` will show pending tests as **SKIPPED** and
implemented tests as **PASSED** or **FAILED**.

---

## What is the difference between unit tests and pytest?

These two terms describe **different things** and are often confused.

### "Unit tests" — a *testing strategy*

A **unit test** is any test that exercises a single, isolated unit of code (a function, a
method, a class) in isolation from the rest of the system.  Unit testing is a *strategy or
approach*, not a specific tool.  The opposite of a unit test is an *integration test*
(testing multiple components together) or an *end-to-end test* (testing the whole system).

The tests in this project are **unit tests** by strategy: each test class targets one
function or class from the source code (e.g. `TestSimpleNNFunctional` targets `SimpleNN`
from `models.py`).

### `unittest` — Python's *built-in testing framework*

Python ships with a module called `unittest` (part of the standard library, no install
needed).  Writing tests with `unittest` looks like this:

```python
import unittest

class TestMyFunction(unittest.TestCase):          # must inherit TestCase
    def test_something(self):
        result = my_function(42)
        self.assertEqual(result, 84)              # assertion via self.assert*()
```

Key characteristics of `unittest`:
- Test classes **must** inherit from `unittest.TestCase`.
- Assertions are made through methods like `self.assertEqual`, `self.assertTrue`, etc.
- Setup and teardown use `setUp()` / `tearDown()` methods.
- Run with `python -m unittest discover`.

### `pytest` — a *third-party testing framework and runner*

`pytest` is a popular third-party library (`pip install pytest`) that can run both its own
test style **and** `unittest.TestCase` tests.  Writing tests with pytest looks like this:

```python
# No base class needed
class TestMyFunction:
    def test_something(self):
        result = my_function(42)
        assert result == 84                       # plain Python assert statement
```

Key characteristics of `pytest`:
- Test classes do **not** inherit from anything.
- Assertions use plain Python `assert` — pytest rewrites them to give detailed failure
  messages automatically.
- Shared setup is handled by *fixtures* in `conftest.py` rather than `setUp()` methods.
- Extra capabilities: parameterised tests, markers, plugins, `pytest.skip()`, `pytest.raises()`.
- Run with `python -m pytest`.

### Summary table

| | `unittest` | `pytest` |
|---|---|---|
| **Type** | Built-in standard-library framework | Third-party framework & runner |
| **Install** | Nothing (ships with Python) | `pip install pytest` |
| **Base class** | Must inherit `unittest.TestCase` | No base class required |
| **Assertions** | `self.assertEqual(a, b)` etc. | Plain `assert a == b` |
| **Shared setup** | `setUp()` / `tearDown()` methods | Fixtures in `conftest.py` |
| **Skip a test** | `@unittest.skip("reason")` | `pytest.skip("reason")` |
| **Pending tests** | No built-in concept | `pytest.skip("Not yet implemented")` |

This project uses **pytest** as the framework and runner.  The tests follow a *unit testing
strategy* because each test class isolates one component of the source code.

---

## How pytest fits the Feature → FR/NFR → Test narrative flow

A well-structured report about this project follows this chain:

```
Feature  →  Functional / Non-Functional Requirements  →  Test Design  →  pytest assertion
```

Each step has a concrete role, and pytest is the mechanism that closes the loop at the end.

---

### Step 1 — Identify the Feature

A *feature* is a behaviour the system must provide.  For example:

> **Feature:** The `LogisticRegression` model must be able to classify input samples into
> probability distributions across the output classes.

---

### Step 2 — Write Functional and Non-Functional Requirements

From that feature, specific requirements are derived:

| Code | Type | Statement |
|---|---|---|
| FR-LR-03 | Functional | `forward()` output shape must be `(n_samples, n_classes)` |
| FR-LR-02 | Functional | `forward()` must return a valid probability distribution (all values in [0, 1], rows sum to 1) |
| FR-LR-04 | Functional | `compute_loss()` must return a non-negative scalar |
| NFR-LR-01 | Non-Functional | `forward()` must complete in under 1 second for typical inputs |
| NFR-LR-02 | Non-Functional | Using the same random seed must produce identical weights (reproducibility) |

Functional Requirements (FR) describe *what* the code does.
Non-Functional Requirements (NFR) describe *quality attributes* — how well it does it.

---

### Step 3 — Design the Test

Each requirement is translated into a test design:

- **Which function** is under test? (`logistic_regression_model.forward()`)
- **What input** is needed? (A small feature matrix `small_X_train` from the fixture)
- **What is the expected result?** (Shape `(20, 3)` for 20 samples and 3 classes)
- **What assertion proves it?** (`assert output.shape == (20, 3)`)

The test class name, test method name, and comment together encode the requirement chain:

```python
class TestLogisticRegressionFunctional:   # ← Feature: LogisticRegression, Type: FR
    # FR-LR-03: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, logistic_regression_model, small_X_train):
        ...                               # ← Requirement code visible in comment
```

---

### Step 4 — The pytest assertion IS the verification

The `assert` statement is the direct, executable translation of the requirement:

```python
class TestLogisticRegressionFunctional:
    # FR-LR-03: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, logistic_regression_model, small_X_train):
        output = logistic_regression_model.forward(small_X_train)
        #                    ↑ call the function under test
        assert output.shape == (small_X_train.shape[0], 3)
        #      ↑ this assert IS the machine-readable version of FR-LR-03
```

---

### Step 5 — pytest runs the assertion and reports the verdict

When `python -m pytest tests/test_models.py` is executed:

| Result | Meaning in the FR/NFR narrative |
|---|---|
| `PASSED` | The implementation **satisfies** this requirement |
| `FAILED` | The implementation **violates** this requirement — pytest prints exactly what was expected and what was received |
| `SKIPPED` | The requirement is specified but the assertion has not yet been written |

A **FAILED** result is the most informative.  For example, if `ComplexNN.backward()` had a
typo (`np.arrange` instead of `np.arange`), `test_train_epoch_returns_loss_and_accuracy`
would fail with:

```
AttributeError: module 'numpy' has no attribute 'arrange'
```

This directly links the runtime error back to the violated requirement (FR-CNN-06), which
links back to the feature.  pytest has *automatically executed* the verification and
produced a traceable failure report.

---

### The complete chain, visualised

```
Feature
│
│  "LogisticRegression must classify inputs into probability distributions"
│
├─► FR-LR-03: forward() output shape is (n_samples, n_classes)
│       │
│       └─► Test design: call forward() with 20-sample input; expect shape (20, 3)
│               │
│               └─► pytest assertion:
│                       output = logistic_regression_model.forward(small_X_train)
│                       assert output.shape == (small_X_train.shape[0], 3)
│                               │
│                               ├── PASSED  → FR-LR-03 satisfied ✓
│                               └── FAILED  → FR-LR-03 violated ✗
│                                             (pytest prints the actual vs expected shape)
│
├─► NFR-LR-01: forward() completes in < 1 second
│       │
│       └─► Test design: time the forward() call; assert elapsed < 1.0
│               │
│               └─► pytest assertion:
│                       start = time.perf_counter()
│                       logistic_regression_model.forward(small_X_train)
│                       assert time.perf_counter() - start < 1.0
│                               │
│                               ├── PASSED  → NFR-LR-01 satisfied ✓
│                               └── FAILED  → NFR-LR-01 violated ✗
│
└─► (further FR/NFR → tests follow the same pattern)
```

---

### Why this matters for a report

In a report, you can:

1. **Reference the FR/NFR code in the test comment** (`# FR-LR-03`) to create a
   traceable link between the requirements section and the testing section.
2. **Show the `pytest` output** (PASSED / FAILED / SKIPPED) as evidence that the
   implementation meets (or does not yet meet) each requirement.
3. **Explain a FAILED test** as evidence that a requirement was identified, a test was
   designed to verify it, and the test caught a real defect — which was then fixed.

The test file structure mirrors the report structure:
- `TestLogisticRegressionFunctional` → the FR section for LogisticRegression
- `TestLogisticRegressionNonFunctional` → the NFR section for LogisticRegression

---

## Running the tests

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run the whole suite
python -m pytest tests/ -v

# Run only the implemented (non-skipped) tests
python -m pytest tests/ -v --ignore=tests/test_database.py --ignore=tests/test_app.py

# Run a single file
python -m pytest tests/test_models.py -v

# Run a single test
python -m pytest tests/test_models.py::TestSimpleNNFunctional::test_forward_output_shape -v
```

## Implementing a pending test

Find the test body you want to implement, remove the `pytest.skip(...)` call, and replace
it with `assert` statements.  For example:

```python
# Before (pending):
def test_forward_output_shape(self, simple_nn_model, small_X_train):
    pytest.skip("Not yet implemented")

# After (implemented):
def test_forward_output_shape(self, simple_nn_model, small_X_train):
    output = simple_nn_model.forward(small_X_train)
    assert output.shape == (small_X_train.shape[0], 3)
```

## File structure

```
tests/
├── README.md              # this file
├── __init__.py            # makes tests/ a Python package
├── conftest.py            # shared pytest fixtures (data, model instances)
├── test_models.py         # LogisticRegression, SimpleNN, ComplexNN  — IMPLEMENTED
├── test_trainer.py        # build_model(), train_model()              — IMPLEMENTED
├── test_dataload.py       # load_dataset_*(), get_dataset_stats()     — IMPLEMENTED
├── test_database.py       # UserProgressTracker (planned class)       — PENDING
└── test_app.py            # Dash app initialisation, layouts, callbacks — PENDING
```

