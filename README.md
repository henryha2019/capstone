# brilliant_automation

## Getting Started

Clone this repository:

```bash
git clone https://github.com/Brilliant-Automation/capstone.git
cd capstone
```

## Create the Environment

Create the environment from the environment.yml file:

```bash
conda env create -f environment.yml
conda activate brilliant-auto-env
```

---

## How to Generate the Proposal Report

Follow these steps to generate the proposal report as a PDF:

### 1. **Run the Preprocessing Script**

Prepare the data required for the report by running the preprocessing script:

```bash
    python model/src/preprocess.py --device "8#Belt Conveyer"
```

This command processes the data for the specified device and prepares it for subsequent analysis.

### 2. **Run the EDA Notebook**

Execute the exploratory data analysis notebook to automatically generate all required plots:

```bash
    jupyter nbconvert --to notebook --execute notebook/eda_conveyer_belt.ipynb
```

This runs all the cells in the notebook and updates it with the generated outputs.

### 3. **Generate the Proposal Report**

Convert the Quarto document into a PDF report using the following command:

```bash
  quarto render docs/proposal.qmd --to pdf
```

The `proposal.pdf` file will be generated and saved in the `docs` directory.

---

## How to Run the Preprocessing Script

The preprocessing script processes raw sensor and ratings data and outputs a merged dataset for further analysis.

### **Supported Devices**

The script currently supports the following devices:
1. `1#High-Temp Fan`
2. `8#Belt Conveyer`
3. `Tube Mill`

### **Usage**

Run the script from the root directory:

``` bash
python model/src/preprocess.py --device "<device_name>" [--data_dir <data_directory>] [--output_dir <output_directory>]
```

#### **Arguments**:

- `--device` (Required): Specify one of the supported device names (e.g. `8#Belt Conveyer`).
- `--data_dir` (Optional): Directory containing raw `.xlsx` files. Defaults to `Data/raw`.
- `--output_dir` (Optional): Directory to save the processed CSV file. Defaults to `Data/process`.

### **Examples**

1. To process data for `8#Belt Conveyer` using default directories:

   ```bash
   python model/src/preprocess.py --device "8#Belt Conveyer"
   ```

   Output:

   ```bash
   Data/process/8#Belt Conveyer_merged.csv
   ```

2. To process data for `Tube Mill` using custom directories:

   ```bash
   python model/src/preprocess.py --device "Tube Mill" --output_dir custom_data/processed
   ```

---

## How to Run the Dashboard

The dashboard currently uses sample data in `data/sample_belt_conveyer.csv` and does not reflect results from our models.

```bash
cd dashboard/src
python -m app
```

Go to `http://127.0.0.1:8050/dashboard/` to view the dashboard.
