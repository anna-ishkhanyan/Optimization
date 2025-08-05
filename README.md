# Optimization Methods

This repository contains Python implementations of various optimization algorithms applied to test functions.

## Files

### 1. `optimization_methods_rosenbrock.py`

Implements optimization algorithms on the Rosenbrock function:

- Newton's Method
- Steepest Descent with line search
- Conjugate Gradient Method (might have issues)

Each method approximates the minimizer of the Rosenbrock function starting from different initial points.

### 2. `steepest_descent_backtracking.py`

Implements the Steepest Descent optimization algorithm with backtracking line search applied to the given function.

The script iteratively updates the solution until the gradient norm is below a tolerance.

---

## Usage

Run the Python scripts in any environment with `numpy` and `scipy` installed.

```bash
python optimization_methods_rosenbrock.py
python steepest_descent_backtracking.py
```
## Dependencies

This repository requires the following Python packages:

- `numpy`  
- `scipy`

You can install them  using pip:

```bash
pip install numpy scipy
