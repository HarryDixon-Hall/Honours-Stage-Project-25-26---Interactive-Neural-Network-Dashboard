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

Running the suite with `python -m pytest tests/` will show all pending tests as **SKIPPED**
(shown as `s` or `SKIPPED` in the output), which is the correct signal that the test exists
and has been *specified* but its assertions have not yet been written.

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

## Running the tests

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run the whole suite — all tests will show as SKIPPED until implemented
python -m pytest tests/ -v

# Run a single file
python -m pytest tests/test_models.py -v

# Run a single test
python -m pytest tests/test_models.py::TestSimpleNNFunctional::test_forward_output_shape -v
```

## Implementing a test

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
├── test_models.py         # LogisticRegression, SimpleNN, ComplexNN
├── test_trainer.py        # build_model(), train_model()
├── test_dataload.py       # load_dataset_*(), get_dataset_stats()
├── test_database.py       # UserProgressTracker (planned class)
└── test_app.py            # Dash app initialisation, layouts, callbacks
```
