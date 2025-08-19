\# Challenge 2 — Multivariate Multi-Step Time-Series Forecasting



Forecast \*\*all variables\*\* for the next \*H\* steps (multi-horizon) from the last \*W\* time steps across all variables. Implemented with an LSTM/Conv1D model and \*\*train-only Min-Max scaling\*\*.



\## TL;DR

\- \*\*Task:\*\* multivariate, multi-step forecasting (supervised framing).

\- \*\*Input:\*\* window of length `W` × `D` features.

\- \*\*Output:\*\* next `H` steps for the same `D` features (or a subset).

\- \*\*Key params:\*\* `window=W`, `stride=S`, `telescope=H`.

\- \*\*Metrics:\*\* MAE per horizon (and overall MAE).



---



\## Repository Structure



📁 results           # It contains the written results

📁 data                   #

📂 src                      #

📂 notebooks                      #





