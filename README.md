# criminalarchetypalanalysis

Archetypal life event analysis for criminal psychoanalysis.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## CLI Usage

The `analysis_cli.py` script provides a simplified entry point for running
clustering and Markov‑chain analysis.

```bash
python analysis_cli.py --type1_dir path/to/type1csvs --type2_csv path/to/type2
```

This script loads the Type1 and optional Type2 data, preprocesses the life
events, performs KMeans clustering and generates a state transition diagram.
