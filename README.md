# Survey-mapping-OCED-Skill-
Code to build OECD-style social and emotional skill measures from the Survey questionnaires from different sources. The pipeline maps survey data self-report items to OECD skill domains.

Code to build OECD-style social and emotional skill measures from the Taiwan Education Panel Survey , junior high student questionnaires, Waves 1 and 2.

The pipeline maps TEPS self-report items to OECD skill domains, recodes them so that higher scores mean higher skill, checks internal consistency, and extracts the first principal component for each domain.

## What it does

1. **Maps** TEPS items to six OECD skill domains: Openness, Task Performance, Engagement with Others, Collaboration, Emotional Regulation, and Compound Skills.
2. **Recodes** items so that a higher score always means a higher skill level. Items on other response formats are recoded explicitly.
3. **Tests reliability** with Cronbach's alpha for each domain and wave.
4. **Extracts PC1 scores** for each domain after filtering weakly correlated items.
5. **Exports** the cleaned item data, so IRT scores can be estimated in R.

## Item-to-skill mapping (LLM-assisted)

Matching survey items to the OECD framework is a content-classification problem. I used an LLM for a first-pass crosswalk, then reviewed every assignment by hand.

- **Model:** [model name and version; date run]
- **Input:** [original item wording and OECD skill definitions]
- **Prompt:** see `crosswalk/prompt.md`
- **Output:** `crosswalk/llm_crosswalk.csv` (item, assigned skill, rationale, final decision)
- **Human review:** [who reviewed; number of items; number changed after review]
- **Agreement:** [LLM vs. final assignment: X of Y items]

The final mapping is hard-coded in the notebook. Items with no clear OECD counterpart, or with a response scale that cannot be combined with the others, are excluded.

## Repository contents

| File | Purpose |
|---|---|
| `OECD_Skills_clean.ipynb` | Main workflow, Waves 1 and 2 |
| `Cronbach_Alpha.py` | Cronbach's alpha and alpha-if-item-deleted |
| `PCA_detail.py` | Item filtering, PC1 scores, loadings, eigenvalues |
| `EFA.py` | Exploratory factor analysis with quartimin rotation, KMO, Bartlett's test |
| `Parallel_Analysis.py` | Parallel analysis for the number of factors |
| `Factor_score_phi.py` | Regression factor scores under oblique rotation |
| `student_data_subset.py` | Snippet: subset of four Wave 1 items |
| `teps_labels_util.py` | Helper for TEPS variable labels |
| `crosswalk/` | LLM prompt and item-to-skill crosswalk |

`EFA.py`, `Parallel_Analysis.py`, and `Factor_score_phi.py` are standalone tools. The notebook imports `EFA` but does not call it.

## Data

TEPS data are **not** included and cannot be redistributed. Apply for access through the TEPS data distributor (Survey Research Data Archive, Academia Sinica).

Then open the configuration cell in the notebook and set:

- `DATA_DIR`: folder with the raw `.dta` files
- `OUT_DIR`: folder for derived files
- the six `W1_*` / `W2_*` file names, to match your local copies

Derived files (`.csv`, `.dta`) contain student IDs. Do not commit them.

## Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
pyreadstat
factor_analyzer
```

Install with `pip install -r requirements.txt`. Keep all `.py` files in the same folder as the notebook, since it imports them by name.

## Notes and limitations

- The scores are **content-based classifications**, not a validated psychometric scale. Several domains have low alpha, notably Engagement with Others and, in Wave 2, Collaboration. Check alpha and loadings before using a domain in analysis.
- Wave 1 and Wave 2 use different items, so scores are not on a common scale.
- Wave 1 item `w1s510` (fighting or conflict with a teacher) enters both Collaboration and Emotional Regulation by design, because the concepts overlap.
- The self-harm screening item (`w1s520`) is deliberately excluded from all scores.

## Citation

[Your name]. *OECD skill scoring pipeline*. GitHub, 2026. https://github.com/Simona-Xuemeng-Wang/Survey-mapping-OCED-Skill-

## License

MIT
