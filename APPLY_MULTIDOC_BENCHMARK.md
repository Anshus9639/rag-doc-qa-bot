# Apply the multi-document benchmark

This package adds a synthetic benchmark without replacing existing application files or the current indexed PDF.

## Copy files into the project

Extract the ZIP into the project root:

`C:\Users\Anshu\rag-doc-qa-bot`

Keep the folder structure. The files should land at these paths:

- `evaluate_multidoc.py`
- `evals\multidoc_cases.json`
- `evals\fixtures\` (eight synthetic `.txt` files)
- `tests\test_multidoc_benchmark.py`
- `README_MULTIDOC_SECTION.md` (temporary README section helper)

The package does not include your existing `data/index`, uploads, `.env`, or any personal documents.

## Run the static checks

From the project root in VS Code PowerShell:

```powershell
python -m pytest -v
```

## Run the benchmark

Retrieval only (embedding model may download on first run; no Ollama generation):

```powershell
python evaluate_multidoc.py --retrieval-only
```

Full evaluation (Ollama must be running and the model in `.env` must be installed):

```powershell
python evaluate_multidoc.py
```

The runner builds its vector store in memory. It does not rewrite or remove `data\index`.

## Update the README

After reviewing the new section, append it to the existing README and delete the helper file:

```powershell
Get-Content .\README_MULTIDOC_SECTION.md | Add-Content .\README.md
Remove-Item .\README_MULTIDOC_SECTION.md
```

Review the appended text before committing.

## Commit only the intended project files

```powershell
git status -sb
git add evaluate_multidoc.py evals/multidoc_cases.json evals/fixtures tests/test_multidoc_benchmark.py README.md
git diff --cached --name-only
git diff --cached --check
git commit -m "Add multi-document RAG evaluation benchmark"
git push origin main
```

Do not stage `.env`, `data/uploads`, `data/index`, or private documents. Run the full benchmark locally before publishing its measured scores in your README or resume.
