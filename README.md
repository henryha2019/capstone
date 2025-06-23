# **Brilliant Automation**

## **1. Getting Started**

### **Clone the Repository**

```bash
git clone https://github.com/Brilliant-Automation/capstone.git
cd capstone
```

### **Create and Activate the Environment**
Create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate brilliant-auto-env
```


---

## **2. Makefile Usage**
> Use the provided `Makefile` to automate the modeling pipeline.

### **Available Commands**
| Command               | Description                                           |
|-----------------------|-------------------------------------------------------|
| `make all`            | Run the full pipeline: download → preprocess → train |
| `make download`       | Only download raw data                                |
| `make preprocess`     | Preprocess data for the specified device              |
| `make features`       | Extract features for the specified device             |
| `make train`          | Train models for the specified device                |
| `make clean`          | Clean temporary files                                 |

### **Using `DEVICE`**
To process a different `DEVICE`, specify it on the command line:
```bash
DEVICE = 8#Belt Conveyer
```
---

## **3. Pipeline Scripts**
Each target in the Makefile maps directly to a Python or shell script.

| Target         | Script                                   | Example                                                          |
|----------------|-----------------------------------------|------------------------------------------------------------------|
| `preprocess`   | `python model/src/preprocess.py`        | `python model/src/preprocess.py --device "8#Belt Conveyer"`      |
| `features`     | `python model/src/feature_engineer.py`  | `python model/src/feature_engineer.py --device "Tube Mill"`      |
| `train`        | `python model/src/model.py`             | `python model/src/model.py --model all --device "Tube Mill"`     |

---

## **4. How to Generate the Proposal Report**

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
    jupyter nbconvert --to notebook --execute notebooks/eda_conveyer_belt.ipynb
```

This runs all the cells in the notebook and updates it with the generated outputs.

### 3. **Generate the Proposal Report**

Convert the Quarto document into a PDF report using the following command:

```bash
  quarto render docs/reports/proposal.qmd --to pdf
```

The `proposal.pdf` file will be generated and saved in the `docs` directory.

---

## **5. How to Run the Preprocessing Script**

The preprocessing script processes raw sensor and ratings data and outputs a merged dataset for further analysis.

###  **Supported Devices**

The script currently supports the following devices:

1. `1#High-Temp Fan`
2. `8#Belt Conveyer`
3. `Tube Mill`

### **Usage**

Run the script from the root directory:

``` bash
python model/src/preprocess.py --device "<device_name>" [--data_dir <data_directory>] [--output_dir <output_directory>]
```

#### **Arguments**

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
   data/processed/8#Belt Conveyer_merged.csv
   model/src/preprocessing.log
   ```

2. To process data for `Tube Mill` using custom directories:

   ```bash
   python model/src/preprocess.py --device "Tube Mill" --output_dir custom_data/processed
   ```

---

## **6. How to Run the Dashboard**

The dashboard currently uses sample data in `data/sample_belt_conveyer.csv` and does not reflect results from our models.

```bash
cd dashboard/src
python -m app
```

Go to `http://127.0.0.1:8050/dashboard/` to view the dashboard.

---

## **7. AWS Deployment**

### Setting up Cron Jobs on AWS EC2

The data pipeline can be automated using cron jobs on AWS EC2. Here's how to set it up:

1. SSH into your EC2 instance:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

2. Set up the cron job:
   ```bash
   crontab -e
   ```

3. Add the following line to run the pipeline every hour (replace paths with your actual paths):
   ```bash
   # Run pipeline at minute 0 of every hour
   0 * * * * PATH=/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin /bin/bash -c '/home/jupyter-ubuntu/capstone/model/scripts/run_pipeline.sh >> /home/jupyter-ubuntu/capstone/model/scripts/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M\%S).log 2>&1'
   ```

    This cron job:
    - Runs at minute 0 of every hour
    - Sets the necessary PATH environment variables
    - Executes the pipeline script
    - Logs output to timestamped files in the logs directory

4. Verify the cron job:
   ```bash
   crontab -l
   ```

5. Check cron service status:
   ```bash
   sudo systemctl status cron
   ```

6. Monitor the logs:
   ```bash
   # View all log files
   ls -l /home/jupyter-ubuntu/capstone/model/scripts/logs/
   
   # View the latest log
   tail -f /home/jupyter-ubuntu/capstone/model/scripts/logs/pipeline_*.log
   ```

If the cron job isn't working as expected:

1. Check cron logs:
   ```bash
   sudo grep CRON /var/log/syslog
   ```

2. Verify script permissions:
   ```bash
   chmod +x /home/jupyter-ubuntu/capstone/model/scripts/run_pipeline.sh
   ```

3. Ensure logs directory exists:
   ```bash
   mkdir -p /home/jupyter-ubuntu/capstone/model/scripts/logs
   chmod 755 /home/jupyter-ubuntu/capstone/model/scripts/logs
   ```

4. Common issues:
   - "Python not found in PATH": Make sure the PATH in the crontab entry includes your Python installation
   - "No MTA installed": This is normal and can be ignored if logs are being written to files
   - Permission denied: Check file and directory permissions
