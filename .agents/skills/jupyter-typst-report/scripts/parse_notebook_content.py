import json
import re
import os


def parse_ipynb_title(notebook_content):
    """
    Parses the notebook content to extract the report title from the first markdown cell.
    Assumes the title is in the format 'Assignment N: [TITLE OF EXPERIMENT]'.
    """
    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            match = re.search(r"#\s*Assignment\s*\d+:\s*(.*)", source)
            if match:
                report_title_str = match.group(1).strip()
                assignment_num_match = re.search(r"Assignment\s*(\d+)", source)
                if assignment_num_match:
                    assignment_num_str = assignment_num_match.group(1).strip()
                else:
                    assignment_num_str = "X"  # Default if not found
                return report_title_str, assignment_num_str
    return "Untitled Report", "X"


def parse_notebook_content(ipynb_path: str):
    """
    Parses a Jupyter Notebook file and extracts structured content in JSON format.
    """
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook_content = json.load(f)

    report_title, assignment_num = parse_ipynb_title(notebook_content)
    notebook_dir = os.path.dirname(ipynb_path)

    parsed_data = {
        "report_title": report_title,
        "assignment_num": assignment_num,
        "notebook_path": ipynb_path,
        "notebook_dir": notebook_dir,
        "cells": [],
    }

    fig_counter = 1  # To track generated figure names

    for i, cell in enumerate(notebook_content["cells"]):
        cell_data = {"cell_type": cell["cell_type"], "source": "".join(cell["source"])}

        if cell["cell_type"] == "code":
            cell_data["execution_count"] = cell.get("execution_count")
            cell_data["outputs"] = []

            for output in cell.get("outputs", []):
                output_data = {}
                if output["output_type"] == "stream" and "text" in output:
                    output_data["type"] = "stream"
                    output_data["content"] = "".join(output["text"])
                    cell_data["outputs"].append(output_data)
                elif (
                    output["output_type"] == "display_data"
                    and "image/png" in output["data"]
                ):
                    # Assume figures are saved as figX.png in the notebook_dir.
                    # The script only needs to provide the *name* for the LLM to use.
                    # The actual saving of the image to disk from base64 is outside this script's scope
                    # unless explicitly requested, and is usually handled by the original notebook execution.

                    output_data["type"] = "image/png"
                    output_data["filename"] = f"fig{fig_counter}.png"

                    # Also extract plt.title from source for LLM to use in caption
                    plot_title_match = re.search(
                        r"plt\.title\([\"'](.*?)[\"']\)", cell_data["source"]
                    )
                    if plot_title_match:
                        output_data["plot_title_from_source"] = plot_title_match.group(
                            1
                        )

                    cell_data["outputs"].append(output_data)
                    fig_counter += 1
                # Add other output types if needed (e.g., execute_result)

        parsed_data["cells"].append(cell_data)

    return json.dumps(parsed_data, indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parse_notebook_content.py <path_to_notebook.ipynb>")
        sys.exit(1)

    ipynb_file_path = sys.argv[1]

    # Ensure the path is absolute
    if not os.path.isabs(ipynb_file_path):
        ipynb_file_path = os.path.abspath(ipynb_file_path)

    if not os.path.exists(ipynb_file_path):
        print(f"Error: Notebook file not found at {ipynb_file_path}")
        sys.exit(1)

    print(parse_notebook_content(ipynb_file_path))
