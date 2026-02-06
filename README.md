# ML Algorithms From Scratch

A collection of various ML algorithms implemented from scratch as part of my Algorithms, AI and ML Laboratory course.

This serves as my dumping ground for everything I have done throughout this course. In each week's folder you will find:
1. Implementation code for the algorithm (`labXX.ipynb`)
2. Associated report (`report.pdf`)
3. Typst code for the reports
4. Matplotlib figures, datasets and other artifacts

## Getting Started
### With `uv` (Recommended)
1. Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) installed
2. Clone the repo
3. Install dependencies
  ```sh
  uv sync
  ```
  4. Run `jupyter`
  ```sh
  uv run jupyter-lab
  ```

### With pip
1. Clone the repo
2. Optionally create a virtual environment and activate it
3. Install dependencies
  ```sh
  pip install .
  ```
  4. Run `jupyter`
  ```sh
  jupyter-lab
  ```

| Topic                                      | Folder Link          |
| ------------------------------------------ | -------------------- |
| Gradient-based Algorithms for Optimization | [Week 1](./week01/)  |
| Regression                                 | [Week 2](./week02/)  |

## Learning from this
I have tried to document the most important learning points in the Jupyter notebook files and the associated reports. 
Some additional points that I didn't find suitable to write in the report are present in the respective week's README.

## Note on accuracy
I am no expert in ML and have taken extensive help from AI while writing the algorithms.
I have given my best effort to get the algorithms correct however there might be issues with correctness and accuracy.
