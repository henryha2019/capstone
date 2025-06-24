.PHONY: all download preprocess features train clean tests proposal-report final-report

# ——— Variables ———
# Default device; override by calling:
#   make preprocess DEVICE="Tube Mill"
#   make train DEVICE="8#Belt Conveyer"
DEVICE ?= 8\#Belt Conveyer

# ——— Default target ———
all: download preprocess features train

# ——— Download step ———
download:
	@echo "🔽 Downloading voltage data..."
	chmod +x ./model/scripts/download_voltage_data.sh
	./model/scripts/download_voltage_data.sh

# ——— Preprocessing ———
preprocess:
	@echo "🧹 Preprocessing for device: $(DEVICE)"
	python model/src/preprocess.py --device "$(DEVICE)"

# ——— Feature extraction ———
features:
	@echo "📐 Extracting features for device: $(DEVICE)"
	python model/src/feature_engineer.py --device "$(DEVICE)"

# ——— Train & tune ———
train:
	@echo "🤖 Training & tuning models for device: $(DEVICE)"
	python model/src/model.py --model all --tune --device "$(DEVICE)"

# ——— Cleanup ———
clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__ *.log *.zip

# --- Tests ---
tests:
	@echo "🧪 Running test cases..."
	pytest -v model/tests/

# Generate the proposal report
.PHONY: proposal-report
proposal-report:
	@echo "Generating proposal report..."
	quarto render docs/reports/proposal.qmd --to pdf
	@echo "Proposal report generated at docs/reports/proposal.pdf"

# Generate the final report
.PHONY: final-report
final-report:
	@echo "Generating final report..."
	quarto render docs/reports/final_report.qmd --to pdf
	@echo "Final report generated at docs/reports/final_report.pdf"

