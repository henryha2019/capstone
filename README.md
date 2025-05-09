# brilliant_automation

### Getting Started

Clone this repository:

```
git clone https://github.com/Brilliant-Automation/capstone.git
cd capstone
```

### Create the Environment
Create the environment from the environment.yml file:

```
conda env create -f environment.yml
conda activate brilliant-auto-env
```
### Run Preprocessing Script

Ensure you are in the root directory and activate your environment:

```
python model/src/preprocess.py
```

### Run the EDA Notebook
The exploratory data analysis for the conveyor belt data is done using a Jupyter notebook:

```
jupyter notebook notebook/eda_conveyer_belt.ipynb
```

This makes sure all plots are generated dynamically by running all cells in order. 

### Generate PDF

To generate the proposal report as a PDF, ensure you have [Quarto](https://quarto.org/) installed. Then run the following command in your terminal from the root directory:

```
quarto render docs/proposal.qmd --to pdf
```

This will generate `proposal.pdf` in the `docs` directory.
