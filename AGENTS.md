# AI Agent Guidelines - ML Algos From Scratch

This repository contains scratch implementations of machine learning algorithms. The following rules and guidelines are mandatory for all AI agents.

## Mandatory Operational Rules

- **Notebook Handling**: **CRITICAL**: Never use standard file tools (`read`, `write`, `edit`) on `.ipynb` files. Always use the **Jupyter MCP connection**.
  - Use `jupyter_read_notebook` to inspect current state.
  - Use `jupyter_insert_cell` or `jupyter_overwrite_cell_source` for modifications.
  - Always execute cells after modification to validate syntax and logic.
- **Dependency Management**: Before importing any package, check `pyproject.toml` under `project.dependencies`. **DO NOT** use or install any packages not listed there.
  - Allowed: `numpy`, `pandas`, `matplotlib`, `cvxpy`, `pytorch`.
  - Forbidden: `scikit-learn`, `tensorflow` (unless specifically added to `pyproject.toml`).
- **Cell Structure**: When implementing a new feature or solving a problem:
  1. Put all definitions (classes, functions, constants) in a single implementation cell.
  2. Put the application, usage, and validation code in one or more **subsequent** cells.
  3. This separation allows for easier debugging and re-use.
- **Reporting**: Use the `jupyter-typst-report` skill to generate or update documentation (`main.typ`) based on notebook execution results. Ensure the notebook is fully executed before running the report generator.
  - **CRITICAL**: When asked to "Discuss", "Conclude", or "Compare", **DO NOT** write these in the notebook (markdown or print). All analysis and conclusions must go in the report file (`main.typ`).
- **Figure ID Generator**: When using `matplotlib`, verify that **cell 2** exactly contains the following code for unique figure IDs:
    ```python
    # -- Generator to get unique figure ids --
    def fig_id_generator():
        i = 1
        while True:
            yield i
            i += 1
    fig_id = fig_id_generator()
    ```
    If not present, insert it at cell 2. This is mandatory only when plotting.
- **Testing/Verification**: Verify that the notebook cells run without any errors.
  - **Verify Notebook**: Execute all cells using Jupyter MCP to ensure the entire pipeline works.
  - **Verify Single Change**: Use `jupyter_execute_cell` via MCP to run specific implementation and validation cells.
  - **Manual Verification**: After modifying an algorithm, run the "usage" cell to visualize results (e.g., plots, metrics). Ensure output matches mathematical expectations.

## MCP Tool Usage Tips

When using Jupyter MCP tools, follow these best practices:

- **Inspecting State**: Use `jupyter_read_notebook` with `response_format="brief"` to get a high-level overview of cell indices before performing modifications.
- **Modifying Cells**: Prefer `jupyter_overwrite_cell_source` when updating existing logic to maintain cell metadata. Use `jupyter_insert_cell` only for adding new steps or verification code.
- **Atomic Operations**: Try to combine related code changes into a single tool call if possible, but keep implementation and usage cells separate as per the mandatory rules.
- **Capturing Output**: Always check the output of `jupyter_execute_cell` to ensure no hidden exceptions occurred during execution.

## Code Style & Conventions

### Imports
- Standard Python imports at the top of the cell.
- Grouping: Standard library, third-party (`numpy`, `pandas`, `matplotlib`), then internal (if any).
- Preferred aliases:
  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  ```

### Formatting & Types
- Follow **PEP 8** for code formatting.
- Use meaningful variable names (snake_case).
- Use Type Hints for function signatures to improve clarity and agent comprehension.
- Algorithms should be implemented from scratch using linear algebra/optimization where possible. Avoid high-level library equivalents like `sklearn.LinearRegression`.
- Use `numpy` vectorization instead of explicit loops where performance or clarity can be improved.

### Error Handling
- Use `try-except` blocks for operations involving complex numerical optimizations (e.g., `cvxpy` solves).
- Validate input data shapes and types before passing them to core algorithm logic.

### Plotting & Figures
- When writing code for plotting figures, you MUST write the code to save the figure in the file system, using the following statement:
    ```python
    plt.savefig(f"fig{next(fig_id)}.png", bbox_inches="tight")
    ```
- This is mandatory whenever `matplotlib` code is used for plotting.

## Project Structure

- `weekXX/`: Organized by course weeks.
  - `labXX.ipynb`: Core implementation.
  - `main.typ`: Typst source for the report.
  - `report.pdf`: Generated report.
  - Data files (`*.data`, `*.csv`): Local datasets used by notebooks.

## Environment Configuration

The environment is configured via `pyproject.toml`. Agents should always check this file first to understand the execution context and available tools.
