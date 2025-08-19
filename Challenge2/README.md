# 📈 Challenge 2 — Multivariate Multi-Step Time-Series Forecasting

Forecast **all variables** for the next *H* steps (multi-horizon) from the last *W* time steps across all variables.  
Implemented with an **LSTM/Conv1D model** and **train-only Min-Max scaling**.

---

## ⚡ TL;DR

- **Task:** Multivariate, multi-step forecasting (supervised framing).
- **Input:** Window of length `W` × `D` features.
- **Output:** Next `H` steps for the same `D` features (or a subset).
- **Key params:** `window=W`, `stride=S`, `telescope=H`.
- **Metrics:** Mean Absolute Error (MAE) per horizon + overall MAE.

---

## 🗂 Repository Structure

```
📁 data/               # Dataset folder (CSV)
📁 src/                # Python source code
📁 notebooks/          # Jupyter notebooks (.ipynb)
📁 results/            # Results (contains the PDF report)
📄 README.md
```

---

## 📊 Data Layout

The dataset is provided as a single **.csv file**, stored inside the `data/` folder.  
It contains all the input features used for the forecasting task.

Example:
```
data/
└── dataset.csv
```

---

## 📑 Results

Inside the `results/` folder you can find a **PDF report** with a detailed summary of the challenge results:

```
results/
└── report.pdf
```

---

## 🙋‍♂️ Author

Giacomo Da Re – [GitHub Profile](https://github.com/Giacomo98DaRe)
