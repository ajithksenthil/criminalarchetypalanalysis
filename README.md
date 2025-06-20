# criminalarchetypalanalysis

Archetypal life event analysis for criminal psychoanalysis.

## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## CLI Usage

The `analysis_cli.py` command combines data loading, preprocessing and clustering. Provide the directory containing Type1 CSV files and, optionally, a Type2 CSV file.

```bash
python analysis_cli.py --type1_dir path/to/type1csvs --type2_csv path/to/type2.csv --n_clusters 5 --diagram state_transition.png [--lexical_impute]
```

This script loads the events, computes embeddings (optionally using lexical imputation), performs K‑Means clustering and generates a state transition diagram.
