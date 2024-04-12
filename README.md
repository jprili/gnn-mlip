# Machine Learning Interaction Potentials (MLIP) with Keras API
This repository is part of my PHYS 310 
(Machine Learning in Physics and Astronomy) final project where I implement
the Behler-Parinello model into the aenet dataset (with some major
simplifications).

To work on this project, clone this repository and activate the
python virtual environment.

## Project Structure
```
.
├── dat/
│   ├── positions/
│   │   └── posXXXX.csv
│   ├── symmetries/
│   │   └── symXXXX.csv
│   └── energies.csv
├── .gitignore
├── mlip.ipynb
├── README.md
├── requirements.txt
├── SymmetryCalculator.py
├── BPNN.py
└── xsf_clean.py
```
`dat` is where all the data is, it is also where files are written for
`xsf_clean.py` and `SymmetryCalculator.py`.
`mlip.ipynb` contains the data analysis done,
and `BPNN.py` contains the class that implements the logic of the model.