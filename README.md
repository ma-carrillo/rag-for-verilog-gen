# rag-for-verilog-gen

Retrieval-Augmented Generation (RAG) pipeline for Verilog code generation using large language models.

By Miguel Ángel Carrillo Cobián for ECSE-689 (HLS) , Prof. Christophe Dubach.

---

## Repository Structure

- **verilog-eval-modifications/**  
  Contains selected modifications from a fork of the [VerilogEval](https://github.com/NVlabs/verilog-eval) framework used to run experiments.

  Key modifications:
  - `scripts/rag_utils.py`: Core implementation of all RAG pipelines (vector, KG, hybrid).  
    *(This file does not exist in the original repository.)*
  - `scripts/sv-generate`: Modified to integrate RAG into the generation pipeline.

  Only these files are included due to the large size of the full repository. The complete modified fork and experimental outputs can be shared if necessary.

- **explore_process_data.ipynb**  
  Notebook for dataset exploration and preprocessing.  
  Includes:
  - VeriRAG dataset analysis
  - RTL-Coder data verification
  - Knowledge graph feature construction

- **RTLCoder-processing.ipynb**  
  Notebook used for preprocessing the RTL-Coder dataset. Originally executed in a Databricks environment; output files are not included in the repository due to their size.
  The original **RTL-Coder dataset** is provided in CSV format. Due to its large size, preprocessing (including embedding generation and knowledge graph feature extraction) was performed using PySpark, and the resulting data is stored as Parquet files.

- **res_analysis.ipynb**  
  Notebook for analyzing experimental results.  
  Due to repository size constraints, generated results are not included, but can be reproduced using the provided pipeline or shared upon request.


- **vector_csv_VERIRAG/**  
  Raw VeriRAG dataset in CSV format (prior to preprocessing).

- **kg_features_parquet/**  
  Preprocessed datasets for knowledge graph (KG) retrieval.  
  Includes feature tables for both VeriRAG and RTL-Coder datasets.

- **vector_parquet/**  
  Preprocessed datasets for vector-based retrieval.  
  Contains embeddings and associated metadata for both datasets.

---

## Running Experiments

### Requirements
- OpenRouter account and API key

### Example: Running Full Benchmark (Vanilla, no RAG) using GPT 3.5 Turbo

```bash
cd ~/verilog-eval-wsl && \
mkdir -p build/openrouter-gpt35-vanilla && \
cd build/openrouter-gpt35-vanilla && \
export OPENROUTER_API_KEY="YOUR_API_KEY" && \
../../configure \
  --with-task=spec-to-rtl \
  --with-model=openai/gpt-3.5-turbo \
  --with-examples=0 \
  --with-samples=1 \
  --with-temperature=0 \
  --with-top-p=0.01 \
  --with-rag=off && \
make SHELL=/bin/bash -j1
````

### RAG Modes

Use the `--with-rag` flag to select retrieval strategy:

* `--with-rag=off` → Vanilla (no retrieval)
* `--with-rag=vector` → Vector-based retrieval
* `--with-rag=kg` → Knowledge graph retrieval
* `--with-rag=hybrid` → Hybrid (KG + vector)

