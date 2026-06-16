# Text2Tabular – Reconstructing Tabular Research Data from Scientific Publications

This repository accompanies the paper _Text2Tabular – Reconstructing Tabular Research Data from Scientific Publications_ and represents an improved, more robust version (bug fixes and more scenarios).

Text2Tabular is a unified NLP-to-synthesis pipeline that automatically reconstructs tabular research datasets directly from natural language descriptions in scientific publications. It combines LLM-driven information extraction with copula-based distribution modeling and constrained MCMC refinement to generate synthetic datasets that preserve the marginal statistics, joint dependencies, and statistical test outcomes reported in the source literature.

This enables rapid access to realistic synthetic data for prototyping, meta-analyses combining studies without shared participant data, reproducibility studies, and privacy-preserving (GDPR/HIPAA) data workflows in fields with limited real-world data.

## Pipeline Overview

The framework follows four stages:

1. **Information extraction** from the publication (marginal statistics, correlations, statistical test results), using a hierarchical chain-of-thought prompting strategy with few-shot examples. PDFs are converted to markdown via **TATR** (extended through `gmft`), **GPT-4.1** extracts dataset characteristics and statistics, and all output is encoded as JSON validated with **Pydantic**. Extraction is conservative: undescribed relationships, missing marginals, and tests referencing undefined variables are skipped.
2. **Structuring** the extracted data into a validated JSON schema.
3. **Analytical construction** of a joint distribution using a **Gaussian copula** (Cholesky decomposition of the correlation matrix). Marginals respect normal/non-normal shapes and prescribed ranges; correlation matrices are populated from coefficients or derived from reported tests; categorical variables are temporarily one-hot encoded for group-specific statistics.
4. **Numerical refinement** via **constrained MCMC** (simulated annealing with Metropolis-Hastings), enforcing category frequencies, value ranges, and reproduction of the extracted statistical tests.

### Mathematical Foundation

The dependency structure is modeled with a Gaussian copula. For a copula with correlation matrix `R`:

```
C_R^Gauss(u_1, ..., u_d) = Φ_R(Φ^{-1}(u_1), ..., Φ^{-1}(u_d))
```

Sampling proceeds via Cholesky decomposition `R = L L^T`, drawing `Z ~ N(0, I)`, applying `Y = L^T Z`, transforming to uniform via `U_i = Φ(Y_i)`, and mapping back through `X_i = Φ_i^{-1}(U_i)`.

The MCMC step uses Metropolis-Hastings acceptance `P(accept) = min(1, exp(-Δ/T))`, where `Δ` is the change in distance between current and target summary statistics and `T` follows a cooling schedule.

## Evaluation

Evaluation is conducted in two stages:

1. **Controlled benchmark datasets** (`iris`, `titanic`, `tips`, `diamonds`) using manually crafted ideal summaries to assess reconstruction fidelity with known ground truth.
2. **Real scientific publications** (seven complex RCTs with accessible raw data) to validate end-to-end robustness.

Reconstruction quality is measured with statistical-similarity metrics (variable coverage, central tendency/dispersion/categorical deviation, KS statistic, Jensen-Shannon and inverse-KL divergence, correlation and statistical-test deviation, outlier and mean-record distance) and ML-utility metrics (feature-importance correlation, and TRTR vs. TSTR gaps in F1 and AUC).

On benchmark datasets, variable coverage and statistical fidelity are nearly perfect and the downstream ML-utility gap is minimal. On real-world publications, coverage drops where variables are omitted or insufficiently described, but key downstream metrics (AUC difference, feature-importance correlation) remain robust. A baseline that directly prompts LLMs to generate tabular data consistently failed to produce valid datasets covering all variables or matching the required statistical properties.

## Repository Structure

```
src/text2tabular/
├── extraction/        # PDF parsing, LLM-based extraction, Pydantic schemas
├── reconstruction/    # Copula sampling, MCMC refinement, generator class
├── evaluation/        # Similarity metrics, ML utility, test suites
└── data/              # Benchmark and publication data
paper/                 # Paper sources, figures, generated result tables
```

## Installation

The project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

An OpenAI API key is required for the extraction stage. Place it in a `.env` file at the repository root:

```
OPENAI_API_KEY=sk-...
```

## Usage

Extraction and reconstruction can be run end-to-end on a PDF, or each stage invoked independently.

```python
from text2tabular.extraction.extractor import extract_summary
from text2tabular.reconstruction.main import SyntheticDataGenerator
from text2tabular.reconstruction.mcmc.mcmc import mcmc_refinement

# 1. Extract structured statistical summary from a publication
summary = extract_summary("path/to/publication.pdf")

# 2. Generate an analytical candidate via copula sampling
generator = SyntheticDataGenerator(parsed_data=summary, seed=42)
candidate = generator.generate_one_instance()

# 3. Refine via constrained MCMC
synthetic_df = mcmc_refinement(candidate, summary)
synthetic_df.to_csv("synthetic_data.csv", index=False)
```

For end-to-end batch extraction over a corpus, see `src/text2tabular/extraction/batch_extract.py`. Worked examples are provided under `src/text2tabular/reconstruction/` and `src/text2tabular/evaluation/`.

## Limitations

- Evaluation is limited to datasets where both publication and raw data are accessible; the current real-data evaluation covers seven RCTs.
- The Gaussian copula assumes linear correlations capture the dependency structure; very strong correlations between categorical variables may not reproduce exactly.
- For poorly described publications, not all variables are recovered automatically; aggregations and ambiguous categorical/ordinal/continuous splits remain a challenge.
- Generated datasets are synthetic and must not be used for clinical decision-making. Biases in the original publication are inherently preserved.

## Acknowledgements

This research has been supported by the German Federal Ministry of Education and Research (BMBF) grant 16IS23069 Software Campus 3.0 (TU München).
