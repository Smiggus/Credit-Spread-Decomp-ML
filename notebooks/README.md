# notebooks/

- **Purpose:** explore data, try ideas, plot results. One notebook per stage, named by stage letter, for example `a_shap_split.ipynb`.
- **Rule:** no production logic lives here. Once an idea works, move it into a `src/` module and import it.
- **Data:** read licensed data through `src.config`, never from a path typed in the notebook.
- **Outputs:** clear cell outputs that show data rows before committing. Totals and charts are fine.
