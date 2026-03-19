---
name: jupyter-typst-report
description: Generate and maintain professional technical reports in Typst format from Jupyter Notebooks (.ipynb) by dynamically generating descriptions, observations, and conclusions based on the notebook's content. Make sure to use this skill whenever the user mentions creating, updating, modifying, or editing a report, generating documentation, or reflecting changes from a Jupyter notebook to a professional document. This includes any request to 're-generate', 'refresh', or 'sync' a report, even if they don't explicitly ask for a 'new' report. This skill leverages the LLM to understand and summarize the notebook's execution and content.
compatibility:
  - python
  - typst
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
            *   **Crucially, DO NOT include the code cell for the figure id generator function.** This function is a utility and not part of the core logic to be reported.
            *   Format the `source` code using `#codly`.
            *   **Outputs:**
                *   If `stream` output is present, format its `content` using `#codly(header: [*Result*], number-format: none)`.
                *   If `image/png` output is present (`filename`), include it using `#figure` and generate a descriptive caption. Prioritize `output.plot_title_from_source` if available, otherwise, infer from the code's likely purpose (e.g., "Impurity Plot", "Accuracy Curves", "Generated Plot").
                *   Generate point-wise "Observations" for any `stream` output or `image/png` plot, summarizing key findings or visual insights.
    *   **Conclusion:** Generate a comprehensive "Conclusion" section (`== Conclusion`) at the end of the report in a *pointwise fashion*. This should summarize the key findings, overall results, and important observations from the entire Jupyter Notebook analysis. Avoid generic placeholders; actively synthesize information from the processed cells.
    *   **Crucially, when creating the report, DO NOT include any text/interpretations from any active chat session/memory; ONLY write contents solely from the notebook contents.**
3.  **Write Final Typst File:** Assemble all the generated Typst content and write it to the specified output file (defaulting to `main.typ` in the same directory as the notebook).
4.  **Compile to PDF:** Execute the `typst compile` command to generate the PDF report.
    *   Default output filename: `report.pdf`
    *   If user specifies a custom filename, use that instead.
    *   Command format: `typst compile main.typ <output_filename>.pdf`
5.  **Open PDF (Optional by Default):** Automatically open the generated PDF using the system's default PDF viewer.
    *   Linux: `xdg-open <output_filename>.pdf`
    *   macOS: `open <output_filename>.pdf`
    *   Windows: `start <output_filename>.pdf`
6.  **Confirm and Provide Output:** Inform the user that the report has been generated and provide the paths to both the `.typ` and `.pdf` files. Mention that the PDF was opened automatically.

## Report Structure and Formatting Rules (as LLM Guidance)

**General Guidance for LLM:**
- Maintain a professional and technical tone.
- Be concise but informative in descriptions and observations.
- Use the provided Typst formatting templates strictly.
- Top-level headings (`=`) should only be used for the main report title. All subsequent major sections should use `==` or lower (e.g., `== Section Title`, `=== Subsection Title`).
- **Writing Style:**
    - Always add a new line separation for headings preceded by paragraphs or `#codly` declarations/code blocks following paragraphs.
    - Always prefer using the auto-numbered `+` for ordered lists rather than manually writing numbers.

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
  == Department of Electrical Engineering \ \
  == Indian Institute of Technology, Kharagpur \ \
  == Algorithms, AI and ML Laboratory (EE22202) \ \
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
-   **Crucially, DO NOT include the code cell for the figure id generator function.** This function is a utility and not part of the core logic to be reported.
-   **Source Code Embedding Rules:**
    - **No Print Statements:** Do not embed lines containing `print()` statements.
    - **Constructor Placeholder:** Do not include a class's `__init__()` if it only contains assignment of member variables. Instead, use this placeholder:
      ```python
      class SomeClass:
          def __init__(self,
                       # <further arguments from the notebook code cell>
                      ):
              # -- SETUP CODE --
      ```
      Fill `SomeClass` and `<further arguments from the notebook code cell>` from the class name and `__init__()` function respectively.
    - **Function Subheadings:** For all important functions in a class (e.g., `.fit()`), create subheadings `===` with an appropriate title. Example:
      ```typ
      === The `.fit()` function
      ```
      Follow this with a **point-wise** description of how that function works.
-   **Code Block Formatting:** Format the processed `cell.source` code:
    ```typ
    #codly(header: [*<Generated Code Block Title>*], number-format: numbering.with("1"))
    ```python
    <processed.cell.source>
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

**Crucially, when creating the report, DO NOT include any text/interpretations from any active chat session/memory; ONLY write contents solely from the notebook contents.**

## Example Usage by User

The user will still initiate the process by providing the notebook path:

```text
Generate a report from my notebook at 'path/to/my_notebook.ipynb'
```
The skill will then execute `parse_notebook_content.py`, process its JSON output, generate `main.typ`, compile it to `report.pdf`, and automatically open the PDF.

### Customizing Output Filename

Users can specify a custom PDF filename:

```text
Generate a report from 'path/to/my_notebook.ipynb' as 'my_report.pdf'
```
If no custom filename is provided, the default `report.pdf` will be used.

### Updating/Editing Existing Reports

Users can request an update or modification to an existing report based on recent notebook changes:

```text
Update the report in week05 to reflect the new accuracy results from the notebook.
Modify the lab03 report; the random forest section needs to be refreshed with the new plot.
Can you edit the report for week04? I've added more analysis in the notebook.
```
When these requests occur, the skill should be invoked to re-process the notebook and update the report accordingly.
