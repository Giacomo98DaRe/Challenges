# Challenges

A collection of small AI/ML/DL projects (notebooks + supporting Python code).  
Each folder `ChallengeX/` contains a runnable notebook and the minimal code to reproduce the results **locally**.

> For project‑specific details (dataset link, model notes, expected results), open the `README.md` inside the challenge you want to run.

---

## Requirements

- **Conda** 25+ (Anaconda / Miniconda / Mambaforge)
- **Python** 3.11 (created from `environment.yml`)
- **JupyterLab** (installed via the environment)

> Everything is CPU‑only by default. No CUDA required.

---

## Repository structure

```
challenges/
  Challenge1/              # check its README.md for challenge‑specific instructions
  Challenge2/              # check its README.md for challenge‑specific instructions
    ...
  environment.yml          # shared conda environment spec
  .gitignore
```

**Data policy**

- Follow the `README.md` inside the challenge for more specific informations.

---

## Quick start (works for every challenge)

Clone and create the environment:

```bash
git clone https://github.com/<your-username>/challenges.git
cd challenges

# create the conda env from the recipe (override the name explicitly)
conda env create -f environment.yml -n challenges-py311
conda activate challenges-py311

# register the Jupyter kernel (once)
python -m ipykernel install --user --name challenges-py311 --display-name "Python (challenges)"
```

Launch JupyterLab and **select the kernel** `Python (challenges)`:

```bash
jupyter lab
```

Open the challenge folder you want (e.g., `Challenge1/notebooks/`) and run the main notebook.

---

## Usage tips

- **One environment, one kernel:** use the kernel `Python (challenges)` for all notebooks here.
- **Relative paths:** notebooks use `pathlib` with paths relative to the challenge folder.

---

## Troubleshooting

- **Kernel not appearing in Jupyter**  
  Re‑install the kernel:
  ```bash
  python -m ipykernel install --user --name challenges-py311 --display-name "Python (challenges)"
  ```
  Then `Kernel → Restart Kernel and Clear All Outputs`.

- **Notebook cannot find data**  
  Ensure you placed files in `ChallengeX/data/dataset/…` or `ChallengeX/data/sample/…`.

- **Conda activation not working in cmd/PowerShell**  
  Initialize your shell and reopen it:
  ```bash
  conda init cmd.exe
  conda init powershell
  ```
  Then `conda activate challenges-py311`.

---

## License

Add a license (e.g., MIT or Apache‑2.0).  
If datasets have their own licenses, follow them accordingly.