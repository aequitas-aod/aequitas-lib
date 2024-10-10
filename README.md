# Aequitas Library

Aequitas is an open-source Python library designed for detecting and mitigating bias in data, algorithms, and social contexts. 
It provides a comprehensive set of tools for developers and stakeholders to assess, repair, and design AI systems with fairness in mind. 
The library supports state-of-the-art bias mitigation techniques and is fully compatible with the existing Python data science ecosystem.

## Project structure 

Overview:
```bash
.
├── aequitas                # main package
│   ├── detection/          # bias detection tools
│   ├── gateway             # remote interaction with other services
│   │   ├── aequitas/       # Aequitas API
│   │   └── AIoD/           # AIoD API
│   ├── mitigation          # bias mitigation tools
│   │   ├── data.py         # pre-processing tools
│   │   └── models.py       # in-processing tools
│   └── tools
│       └── data_manip.py   # utilities for data manipulation
├── datasets/               # datasets used for examples
├── examples/               # examples of usage of the library (JuPyter notebooks)
├── LICENSE                 # Apache 2.0 License file
├── package.json
├── package-lock.json
├── pyproject.toml
├── README.md
├── release.config.js
├── renovate.json
├── requirements-dev.txt
├── requirements.txt
└── setup.py
```

## Installation

To install the library, you can use pip:

```bash
git clone https://github.com/aequitas-aod/aequitas-lib
cd aequitas-lib
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# play with examples in the examples folder
```
