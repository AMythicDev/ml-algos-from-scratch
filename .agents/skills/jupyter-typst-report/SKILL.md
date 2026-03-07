---
name: jupyter-typst-report
description: Generate professional technical reports in Typst format from Jupyter Notebooks (.ipynb) by dynamically generating descriptions, observations, and conclusions based on the notebook's content. Use this skill whenever the user asks to create a report, generate documentation, or convert a Jupyter notebook to a professional document, especially when Typst or high-quality technical reports are mentioned. This skill leverages the LLM to understand and summarize the notebook's execution and content.
compatibility:
  - python
---

# Jupyter to Typst Report Generator (LLM-Enhanced)

This skill automates the creation of comprehensive technical reports in the Typst typesetting language directly from Jupyter Notebook (.ipynb) files. It extracts raw notebook content, and then the LLM dynamically analyzes and formats this content into a professional, well-structured document, including descriptive text, observations, and conclusions.

## Workflow

1.  **Parse Notebook Content:** Execute the `scripts/parse_notebook_content.py` script, passing the `.ipynb` file path. This script will output a JSON object containing the structured content of the notebook (cells, source, outputs, etc.).
    *   Example: `python jupyter-typst-report/scripts/parse_notebook_content.py <path_to_notebook.ipynb>`
2.  **LLM Generates Typst Report:** The LLM will consume the JSON output from the parsing script. It will then:
    *   **Preamble:** Construct the Typst preamble using the extracted `report_title` and `assignment_num`.
    *   **Cell-by-Cell Processing:** Iterate through the `cells` array in the JSON output.
        *   **Markdown Cells:** Directly include the `source` content, converting markdown headings (e.g., `# Header`) to Typst headings (e.g., `= Header`).
        *   **Code Cells:**
            *   Generate a suitable subheading (e.g., "Data Loading and Preprocessing", "Model Definition", "Experiment Execution") and a concise description of the code's purpose based on its `source` code and any preceding markdown cell context.
            *   Format the `source` code using `#codly`.
            *   **Outputs:**
                *   If `stream` output is present, format its `content` using `#codly(header: [*Result*], number-format: none)`.
                *   If `image/png` output is present (`filename`), include it using `#figure` and generate a descriptive caption. Prioritize `output.plot_title_from_source` if available, otherwise, infer from the code's likely purpose (e.g., "Impurity Plot", "Accuracy Curves", "Generated Plot").
                *   Generate point-wise "Observations" for any `stream` output or `image/png` plot, summarizing key findings or visual insights.
    *   **Conclusion:** Generate a comprehensive "Conclusion" section (`== Conclusion`) at the end of the report in a *pointwise fashion*. This should summarize the key findings, overall results, and important observations from the entire Jupyter Notebook analysis. Avoid generic placeholders; actively synthesize information from the processed cells.
3.  **Write Final Typst File:** Assemble all the generated Typst content and write it to the specified output file (defaulting to `main.typ` in the same directory as the notebook).
4.  **Confirm and Provide Output:** Inform the user that the report has been generated and provide the path to the `.typ` file.

## Report Structure and Formatting Rules (as LLM Guidance)

**General Guidance for LLM:**
- Maintain a professional and technical tone.
- Be concise but informative in descriptions and observations.
- Use the provided Typst formatting templates strictly.
- Top-level headings (`=`) should only be used for the main report title. All subsequent major sections should use `==` or lower (e.g., `== Section Title`, `=== Subsection Title`).

### 1. Preamble

Construct the report preamble using the extracted `report_title` and `assignment_num`.

```typ
#set par(leading: 0.55em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#codly(languages: codly-languages, display-icon: false, display-name: false, breakable: true)

#align(center + horizon)[
  == Department of Electrical Engineering \
  == Indian Institute of Technology, Kharagpur \
  == Algorithms, AI and ML Laboratory (EE22202) \
  == Spring, 2025-26 \
  \
  = Report <assignment_num>: <report_title>
  \
  == Name: Arijit Dey \
  == Roll No: 24IE10001
]

#pagebreak()

#align(center)[= <report_title_main_part>]

#set heading(numbering: (..nums) => {
  if nums.pos().len() > 1 {
    numbering("1.1", ..nums.pos().slice(1, none))
  }
})
```
`<report_title_main_part>` refers to the part of `report_title` after the colon.

### 2. Cell Processing (LLM's Task)

#### Markdown Cells

-   Read `cell.source`.
-   If it's the first cell and matches `# Assignment N: [TITLE]`, skip it (it's in the preamble).
-   Convert markdown headings (`#`, `##`, etc.) to Typst headings (`=`, `==`, etc.).
-   Include all other markdown text directly.

#### Code Cells

-   **Generate Subheading and Description:** Based on the `cell.source` code, generate a suitable subheading (e.g., "Data Loading and Preprocessing", "Model Definition", "Experiment Execution") and a brief descriptive paragraph explaining the code's purpose. Consider the overall flow of the notebook.
-   **Code Block Formatting:** Format the `cell.source` code:
    ```typ
    #codly(header: [*<Generated Code Block Title>*], number-format: numbering.with("1"))
    ```python
    <cell.source>
    ```
-   **Process Outputs (`cell.outputs` array):**
    *   **Stream Output:** If `output.type == "stream"`, format its `output.content`:
        ```typ
        #codly(header: [*Result*], number-format: none)
        ```
        <output.content>
        ```
    *   **Image Output:** If `output.type == "image/png"`, include the image (`output.filename`) and generate a descriptive caption. Prioritize `output.plot_title_from_source` if available, otherwise, infer from the code's likely purpose (e.g., "Impurity Plot", "Accuracy Curves", "Generated Plot").
    *   **Generate Observations:** If there are any `stream` outputs or `image/png` plots, generate a point-wise "Observations" section immediately following them, summarizing key results, patterns, or visual insights.

### 3. Conclusion (LLM's Task)

Generate a comprehensive "Conclusion" section (`== Conclusion`) at the end of the report in a *pointwise fashion*. This should summarize the key findings, overall results, and important observations from the entire Jupyter Notebook analysis. Avoid generic placeholders; actively synthesize information from the processed cells.

## Example Usage by User

The user will still initiate the process by providing the notebook path:

```text
Generate a report from my notebook at 'path/to/my_notebook.ipynb'
```
The skill will then execute `parse_notebook_content.py`, process its JSON output, and generate `main.typ`.
