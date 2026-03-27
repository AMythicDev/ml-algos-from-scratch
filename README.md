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

## List of Topics

| Topic                                                            | Folder Link         |
| ---------------------------------------------------------------- | ------------------- |
| Gradient-based Algorithms for Optimization                       | [Week 1](./week01/) |
| Regression                                                       | [Week 2](./week02/) |
| Support Vector Machines                                          | [Week 3](./week03/) |
| Decision Trees                                                   | [Week 4](./week04/) |
| Random Forests                                                   | [Week 5](./week05/) |
| Artificial Neural Networks                                       | [Week 6](./week06/) |
| Fashion MNIST Classification using Convolutional Neural Networks | [Week 9](./week09/) |

## Connecting Jupyter Notebook With Your AI
Jupyter Lab can be integrated with your AI model of choice using the [MCP protocol](https://modelcontextprotocol.io/docs/getting-started/intro). Simply start Jupyter Lab with `just` using

```
just jupyter
```

and connect with your MCP client of choice.

Make sure to set the environment variables as mentioned in [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server#-getting-started) docs.

Support for [OpenCode](https://opencode.ai/) is already configured in the repo since that's what I use.

## Report Generation
There's a agent skill `jupyter-typst-report` which can write a typst report for a given notebook. Compatible with [Claude Code](https://claude.com/product/claude-code), [Codex](https://openai.com/codex/), [GitHub Copilot](https://github.com/features/copilot), [Gemini CLI](https://geminicli.com/), OpenCode, or any other AI agent that support [skills](https://skills.sh) protocol.

## Learning from this
I have tried to document the most important learning points in the Jupyter notebook files and the associated reports. 
Some additional points that I didn't find suitable to write in the report are present in the respective week's README.

## Note on accuracy
I am no expert in ML and have taken extensive help from AI while writing the algorithms.
I have given my best effort to get the algorithms correct however there might be issues with correctness and accuracy.
