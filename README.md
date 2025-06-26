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

Data can be obtained via one of the following methods:

- **Google Drive**: [Download the dataset ZIP](https://drive.google.com/file/d/1dsNqNYfvnSS6W5rAwDHOULK0m53Artsp/view?usp=drive_link)
- **S3 Bucket**: `s3://brilliant-automation/capstone/data/voltage/`

After downloading, unzip into the project root:

```bash
unzip voltage_data.zip -d data/voltage/
```

#### Runs Pipeline Locally

```bash
make all
```

Runs the complete pipeline in sequence:

1. **Preprocess data** (`make preprocess`)
2. **Extract features** (`make features`)
3. **Train models** (`make train`)
4. **Run tests** (`make tests`)
5. **Launch dashboard** (`make dashboard`)

### **Project Structure**

A quick overview of the repository layout:

```
├── dashboard/src/            # Dash app and tests
├── data/                     # Data storage
│   ├── raw/                  # Raw sensor and rating files
│   ├── processed/            # Preprocessed CSV outputs
│   ├── features/             # Extracted feature CSVs
│   ├── voltage/              # Unzipped voltage data
│   └── rar_files/            # Archived raw data
├── docs/                     # Documentation and report sources
│   ├── images/               # Diagrams and figures
│   └── reports/              # Quarto QMD and generated PDFs
├── model/                    # Modeling scripts and results
│   ├── src/                  # Preprocessing, feature, and model code
│   ├── tests/                # Unit tests for modeling pipeline
│   └── archive/              # Legacy scripts
│   └── utils/                # EDA helper module
│   └── scripts/              # Pipeline scripts
│   └── results/              # Model outputs
├── notebooks/                # EDA notebooks for each device
├── Makefile                  # Pipeline orchestration
├── environment.yml           # Conda environment spec
└── README.md                 # Project README
```

---

## **2. Makefile Usage**

> Use the provided `Makefile` to automate the modeling pipeline.

### Available Commands

| Command                        | Description                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| `make all`                     | Run the full pipeline: download → preprocess → features → train → tests → dashboard |
| `make preprocess`              | Preprocess data for the specified device                                            |
| `make features`                | Extract features for the specified device                                           |
| `make train`                   | Train and tune models for the specified device                                      |
| `make tests`                   | Run all unit tests using `pytest`                                                   |
| `make clean`                   | Remove temporary and generated files                                                |
| `make dashboard`               | Preprocess missing devices and launch the dashboard server                          |
| `make proposal-report`         | Generate the proposal report (PDF)                                                  |
| `make technical-report`        | Generate the technical report (PDF)                                                 |
| `make final-report`            | Generate the final report (PDF)                                                     |
| `make reports`                 | Run all report targets: proposal, technical, and final                              |
| `make notebooks`               | Execute and render EDA Jupyter notebooks for each device                            |
| `make archive-feature-eng-rar` | Run archived feature extraction script                                              |
| `make archive-model`           | Run archived model script                                                           |
| `make archive-rnn`             | Run archived RNN model script                                                       |
| `make archive`                 | Run all archived scripts                                                            |

---

### Using `DEVICE`

To target a specific device, set the `DEVICE` variable:

```bash
make preprocess DEVICE="Tube Mill"
```
The script currently supports the following devices:

1. `1#High-Temp Fan`
2. `8#Belt Conveyer`
3. `Tube Mill`

---

## **3. Pipeline Scripts**
Each `make` target maps directly to a script or command.

| Target             | Script/Command                                                           | Example                                  |
| ------------------ | ------------------------------------------------------------------------ | ---------------------------------------- |
| `preprocess`       | `python model/src/preprocess.py --device "$(DEVICE)"`                    | `make preprocess DEVICE="Tube Mill"`     |
| `features`         | `python model/src/feature_engineer.py --device "$(DEVICE)"`              | `make features DEVICE="8#Belt Conveyer"` |
| `train`            | `python model/src/model.py --model all --tune --device "$(DEVICE)"`      | `make train DEVICE="8#Belt Conveyer"`    |
| `tests`            | `pytest -v model/tests/ dashboard/src/tests/`                            | `make tests`                             |
| `clean`            | Remove caches, logs, and temporary files                                 | `make clean`                             |
| `dashboard`        | Preprocess any missing devices, then `cd dashboard/src && python -m app` | `make dashboard`                         |
| `proposal-report`  | `quarto render docs/reports/proposal.qmd --to pdf`                       | `make proposal-report`                   |
| `technical-report` | `quarto render docs/reports/technical_report.qmd --to pdf`               | `make technical-report`                  |
| `final-report`     | `quarto render docs/reports/final_report.qmd --to pdf`                   | `make final-report`                      |
| `notebooks`        | `jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace`  | `make notebooks`                         |
| `archive-*`        | Run archived scripts under `model/archive/`                              | `make archive`                           |

---

## **4. How to Generate the Proposal Report**

Use the Makefile in three simple steps:

```bash
make preprocess DEVICE="<device_name>"    # Prepare data for the chosen device
make notebooks                              # Execute all EDA notebooks and generate plots
make proposal-report                        # Render the proposal QMD into proposal.pdf
```

The `proposal.pdf` file will be generated and saved in the `docs` directory.

---

## **5. How to Generate the Final Report**

Convert the Quarto document into a PDF report using the following command:

```bash
make final-report
```

The `final_report.pdf` file will be generated and saved in the `docs` directory.

---

## **6.1. How to Run the Preprocessing Script**

The preprocessing script processes raw sensor and ratings data and outputs a merged dataset for further analysis.

### **Supported Devices**

The script currently supports:

1. `1#High-Temp Fan`
2. `8#Belt Conveyer`
3. `Tube Mill`

### **Usage**

From the project root, run:

```bash
python model/src/preprocess.py --device "<device_name>" [--data_dir <data_directory>] [--output_dir <output_directory>]
```

#### **Arguments**

- `--device` (Required): One of the supported device names.
- `--data_dir` (Optional): Directory containing raw `.xlsx` files (default: `data/raw`).
- `--output_dir` (Optional): Directory for the processed CSV (default: `data/processed`).

### **Examples**

1. Default directories for `8#Belt Conveyer`:

   ```bash
   python model/src/preprocess.py --device "8#Belt Conveyer"
   ```

   Outputs:

   ```bash
   data/processed/8#Belt Conveyer_merged.csv
   model/src/preprocessing.log
   ```

2. Custom output directory for `Tube Mill`:

   ```bash
   python model/src/preprocess.py --device "Tube Mill" --output_dir custom_data/processed
   ```

---

## **6.2. How to Run Feature Extraction**

The feature extraction script computes DSP metrics and aggregates them with ratings to prepare the analysis-ready dataset.

### **Supported Devices**

The script accepts the same devices as preprocessing:

1. `1#High-Temp Fan`
2. `8#Belt Conveyer`
3. `Tube Mill`

### **Usage**

From the project root, run:

```bash
python model/src/feature_engineer.py --device "<device_name>" [--input_dir <processed_data_directory>] [--output_file <output_csv_path>]
```

#### **Arguments**

- `--device` (Required): Device name matching the processed data file.
- `--input_dir` (Optional): Directory containing preprocessed CSVs (default: `data/processed`).
- `--output_file` (Optional): Path for the feature CSV output (default: `data/features/<device>_features.csv`).

### **Examples**

1. Default input/output for `8#Belt Conveyer`:

   ```bash
   python model/src/feature_engineer.py --device "8#Belt Conveyer"
   ```

   Outputs:

   ```bash
   data/features/8#Belt Conveyer_features.csv
   model/src/feature_engineer.log
   ```

2. Custom input directory for `8#Belt Conveyer`:

   ```bash
   python model/src/feature_engineer.py --device "8#Belt Conveyer" --input_dir custom_data/processed --output_file custom_data/features/tube_mill.csv
   ```

---

## **6.3. How to Run Model Training**

The model training script fits and tunes selected algorithms on the feature dataset.

### **Supported Devices & Models**

- **Devices**: Same three devices as above.
- **Models**: Specify `--model all` or a comma-separated list of keys (e.g., `ridge,rf,xgb,rnn`).

### **Usage**

From the project root, run:

```bash
python model/src/model.py --device "<device_name>" --model <model_list> [--tune] [--output_dir <model_dir>]
```

#### **Arguments**

- `--device` (Required): Device name whose feature CSV is used.
- `--model` (Required): Comma-separated models or `all`.
- `--tune` (Optional): Enable hyperparameter tuning via cross-validation.
- `--output_dir` (Optional): Directory to save trained models and metrics (default: `models/`).

### **Examples**

1. Train all models with tuning for `8#Belt Conveyer`:

   ```bash
   python model/src/model.py --device "8#Belt Conveyer" --model all --tune
   ```

   Outputs:

   ```bash
   results/models/8#Belt Conveyer/                            # trained model files
   results/metrics/8#Belt Conveyer/metrics_summary.csv        # performance metrics
   ```

2. Train only Ridge and Random Forest without tuning:

   ```bash
   python model/src/model.py --device "8#Belt Conveyer" --model ridge,rf
   ```

---

## **7. How to Run Unit Tests**

The project includes comprehensive unit tests for the data processing pipeline. Tests are located in the `model/tests/` directory and `dashboard/src/tests/` and cover preprocessing, feature engineering, model functionality, and key dashboard scripts.

```bash
make tests
```

---

## **8. How to Run the Dashboard**

```bash
make dashboard
```

Go to [http://127.0.0.1:8050/dashboard/](http://127.0.0.1:8050/dashboard/) to view the dashboard.

<img src="data/Dashboard.gif" alt="Dashboard" width="70%" />

---

## **9. AWS Data Pipeline**

### Architecture Overview

Our AWS data pipeline provides an end-to-end solution for automated equipment health monitoring, leveraging cloud services for scalable data processing and machine learning model deployment.

#### **Technical Architecture**

![AWS Data Pipeline Architecture](docs/images/final_report_figures/AWS_Data_Pipeline.png)

**Key Components:**
- **Storage Layer**: AWS S3 for centralized data repository
- **Processing Layer**: AWS EC2 for data processing, feature engineering, and ML training
- **Application Layer**: Flask dashboard and automated scheduling

#### **AWS Services Used**

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **Amazon S3** | Centralized data storage | Bucket: `brilliant-automation-capstone` |
| **Amazon EC2** | Compute infrastructure | Instance for processing and dashboard hosting |
| **AWS IAM** | Access management | Secure S3 and EC2 permissions |

### **Data Flow Architecture**

#### **S3 Bucket Structure**
```
brilliant-automation-capstone/
├── raw/                    # Raw sensor data (.xlsx files)
├── voltage/               # Compressed voltage data (.7z files)
│   ├── 20250401-8#Belt Conveyer/    # Extracted JSON files
│   ├── 20250402-8#Belt Conveyer/    
│   └── ...
├── processed/             # Processed datasets (.csv files)
│   ├── 8#Belt Conveyer_merged.csv
│   ├── 8#Belt Conveyer_full_features.csv
│   └── ...
└── results/               # ML results and artifacts
    ├── models/            # Trained models (.pkl files)
    ├── metrics/           # Performance metrics (.csv files)
    └── plots/             # Prediction visualizations (.png files)
```

#### **Pipeline Stages**

1. **Data Ingestion**: Raw sensor data uploaded to S3 `raw/` folder
2. **Preprocessing**: Sensor and rating data merged and synchronized
3. **Feature Engineering**: DSP metrics extracted from voltage data
4. **Model Training**: Multiple ML models trained with hyperparameter tuning
5. **Results Storage**: Models, metrics, and plots saved to S3 `results/` folder

### **Complete Runbook**

#### **Prerequisites**

1. **AWS Account Setup**
   - Valid AWS account with appropriate permissions
   - Obtain the required credentials from Brilliant Automation
   - AWS CLI configured with credentials

#### **Initial Setup**

##### **1. AWS Console Access**

1. **Access AWS Console**
   - Navigate to: [https://073680586744.signin.aws.amazon.com/console](https://073680586744.signin.aws.amazon.com/console)
   - Obtain username and password from Brilliant Automation
   - Login to access the AWS management console

2. **Obtain EC2 Instance Information**
   - Navigate to EC2 service in the AWS Console
   - Find the running instance and note the public IP address
   - Obtain the `ec2-access-key.pem` file from Brilliant Automation

3. **Access S3 Bucket**
   - Navigate to S3 service in the AWS Console
   - Search for and select the `brilliant-automation-capstone` bucket
   - Explore the folder structure:
     - `raw/` - Raw sensor data (.xlsx files)
     - `voltage/` - Compressed voltage data (.7z files) and extracted JSON files
     - `processed/` - Processed datasets (.csv files)
     - `results/` - ML results (models, metrics, plots)
   - Use the bucket for uploading new data or downloading results

##### **2. EC2 Instance Login**

```bash
# Set correct permissions for the key file
chmod 400 ~/Downloads/ec2-access-key.pem

# SSH into the EC2 instance
ssh -i ~/Downloads/ec2-access-key.pem ubuntu@<ip_address>

# Replace <ip_address> with the actual IP address from AWS Console
# Example: ssh -i ~/Downloads/ec2-access-key.pem ubuntu@54.175.183.157
```

**Note**: Obtain the `ec2-access-key.pem` file and IP address from Brilliant Automation team.

##### **3. EC2 Project Setup**

**Note**: The capstone project folder is already located at `/home/jupyter-ubuntu/capstone` on the EC2 instance.

1. **GitHub Access Setup**
   ```bash
   # Navigate to the project directory
   cd /home/jupyter-ubuntu/capstone
   
   # Generate SSH key pair for GitHub access
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default file location
   # Set a passphrase (optional but recommended)
   
   # Display the public key
   cat ~/.ssh/id_ed25519.pub
   ```

2. **Add SSH Key to GitHub Account**
   - Copy the output of the `cat ~/.ssh/id_ed25519.pub` command
   - Go to [GitHub SSH Settings](https://github.com/settings/keys)
   - Click "New SSH key"
   - Paste the public key and give it a descriptive title
   - Click "Add SSH key"

3. **Configure Git and Test Access**
   ```bash
   # Configure git (replace with your details)
   git config --global user.name "Your Name"
   git config --global user.email "your_email@example.com"
   
   # Test SSH connection to GitHub
   ssh -T git@github.com
   
   # Update remote URL to use SSH (if needed)
   git remote set-url origin git@github.com:Brilliant-Automation/capstone.git
   
   # Verify access by pulling latest changes
   git pull origin main
   ```

4. **Environment Setup**
   ```bash
   # Create conda environment (if not already created)
   conda env create -f environment.yml
   conda activate brilliant-auto-env

**Note**: Ensure you have the necessary permissions to access the [Brilliant-Automation/capstone](https://github.com/Brilliant-Automation/capstone) repository.

##### **4. EC2 JupyterHub Setup**

Access JupyterHub on browser:
- **URL**: http://<ip_address> (e.g., [http://54.175.183.157](http://54.175.183.157))
- **Username**: `ubuntu`
- **Password**: Obtain from Brilliant Automation team

**Note**: The JupyterHub interface provides direct access to the project environment and allows you to run notebooks, edit code, and manage files through a web browser.


#### **Pipeline Execution**

##### **Manual Execution**

1. **Run Complete Pipeline**
   ```bash
   # Make script executable
   chmod +x model/scripts/run_pipeline.sh
   
   # Execute full pipeline
   ./model/scripts/run_pipeline.sh
   ```

2. **Individual Pipeline Steps**
   ```bash
   # Step 1: Preprocessing (all devices)
   python model/src/preprocess.py --device "8#Belt Conveyer" --aws
   python model/src/preprocess.py --device "1#High-Temp Fan" --aws
   python model/src/preprocess.py --device "Tube Mill" --aws
   
   # Step 2: Feature Engineering
   python model/src/feature_engineer.py --device "8#Belt Conveyer" --aws
   
   # Step 3: Model Training (all models)
   python model/src/model.py --model Baseline --device "8#Belt Conveyer" --aws --tune
   python model/src/model.py --model Ridge --device "8#Belt Conveyer" --aws --tune
   # ... (continues for all 7 models)
   ```

##### **Automated Execution with Cron**

The cron job automatically runs the complete data pipeline every 6 hours, executing the following programs in sequence:
- **Data Preprocessing**: `preprocess.py` for all 3 devices (1#High-Temp Fan, 8#Belt Conveyer, Tube Mill)
- **Feature Engineering**: `feature_engineer.py` for 8#Belt Conveyer
- **Model Training**: `model.py` for all 7 models (Baseline, Ridge, PolyRidgeDegree2, RandomForest, XGBoost, SVR, RuleTree)

1. **Setup Cron Job**
   ```bash
   # Edit crontab
   crontab -e
   
   # Add daily pipeline execution (runs every 6 hours)
   0 */6 * * * PATH=/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin /bin/bash -c '/home/jupyter-ubuntu/capstone/model/scripts/run_pipeline.sh >> /home/jupyter-ubuntu/capstone/model/scripts/logs/pipeline_$(date +\%Y\%m\%d_\%H\%M\%S).log 2>&1'
   ```

2. **Logging Directory**
   The logs of the Cron job is currently located in `/home/jupyter-ubuntu/capstone/model/scripts/logs`.

#### **Dashboard Deployment**

```bash
# Navigate to dashboard directory
cd dashboard/src

# Install dashboard dependencies (if not already installed)
conda activate brilliant-auto-env

# Run dashboard with AWS integration
python app.py --aws
```

After starting the dashboard, you can view it at: **http://<ip_address>:8050/dashboard/**

Example: [http://54.175.183.157:8050/dashboard/](http://54.175.183.157:8050/dashboard/)

**To run in background:**
```bash
# Run dashboard in background (keeps running even after closing terminal)
nohup python -m app --aws > /dev/null 2>&1 &

# To stop the dashboard later:
ps aux | grep python
kill <PID>
```

**Note**: Use `nohup` if you want to keep the dashboard running even after you close your AWS terminal session.
