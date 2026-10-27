
## Multi-document evaluation benchmark

The repository includes a separate, reproducible benchmark corpus made of synthetic policy and vendor-agreement text files under `evals/fixtures/`. These are test fixtures, not actual company policies or legal advice. The benchmark builds an isolated in-memory FAISS store and does not modify the documents or vectors under `data/index/`.

Install the project's normal dependencies first. To measure global retrieval across the fixture corpus without calling the generation model, run:

```powershell
python evaluate_multidoc.py --retrieval-only
```

To run retrieval plus answer, source/section citation, evidence-snippet, and abstention checks, make sure Ollama is running and the model configured in `.env` is available, then run:

```powershell
python evaluate_multidoc.py
```

The dataset contains answerable questions as well as questions whose requested facts are deliberately absent. The script reports Hit@1, Hit@3 and MRR for annotated answerable retrieval targets, and checks answer terms, citations to the expected source and section, evidence snippets, and exact abstention for missing facts. The static benchmark checks run with the normal test suite via `python -m pytest -v` and do not require Ollama.

Treat the results as a small, manually labelled development benchmark, not a statistically representative measurement of general RAG accuracy. When adding test cases, independently review the expected source, section, answer groups, and evidence groups to avoid tuning the benchmark to model outputs.
