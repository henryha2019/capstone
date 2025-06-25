.PHONY: all download preprocess features train clean tests dashboard 

# ——— Variables ———
# Default device; override by calling:
#   make preprocess DEVICE="Tube Mill"
#   make train DEVICE="8#Belt Conveyer"
DEVICE ?= 8\#Belt Conveyer

# ——— Default target ———
all: download preprocess features train tests dashboard

# ——— Download step ———
download:
	@echo "🔽 Downloading voltage data..."
	@chmod +x ./model/scripts/download_voltage_data.sh
	@./model/scripts/download_voltage_data.sh

# ——— Preprocessing ———
preprocess:
	@echo "🧹 Preprocessing for device: $(DEVICE)"
	@python model/src/preprocess.py --device "$(DEVICE)"

# ——— Feature extraction ———
features:
	@echo "📐 Extracting features for device: $(DEVICE)"
	@python model/src/feature_engineer.py --device "$(DEVICE)"

# ——— Train & tune ———
train:
	@echo "🤖 Training & tuning models for device: $(DEVICE)"
	@python model/src/model.py --model all --tune --device "$(DEVICE)"

# ——— Tests ———
tests:
	@echo "🧪 Running test cases..."
	@pytest -v model/tests/ dashboard/src/tests/

# ——— Cleanup ———
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ *.log *.zip

# ——— Dashboard ———
dashboard:
	@echo "🚀 Launching dashboard..."
	@for dev in '8#Belt Conveyer' '1#High-Temp Fan' 'Tube Mill'; do \
		file="data/processed/$${dev}_merged.csv"; \
		if [ ! -f "$$file" ]; then \
			echo "⚙️ Preprocessing missing $$dev"; \
			$(MAKE) preprocess DEVICE="$$dev"; \
		fi; \
	done
	@echo "➡️ Starting dashboard server"
	@(cd dashboard/src && python -m app)

# ——— Reports ———
.PHONY: reports proposal-report technical-report final-report
reports: proposal-report technical-report final-report

proposal-report:
	@echo "📄 Generating proposal report..."
	@quarto render docs/reports/proposal.qmd --to pdf
	@echo "Proposal report generated at docs/reports/proposal.pdf"

technical-report:
	@echo "📄 Generating technical report..."
	@quarto render docs/reports/technical_report.qmd --to pdf
	@echo "Technical report generated at docs/reports/technical_report.pdf"

final-report:
	@echo "📄 Generating final report..."
	@quarto render docs/reports/final_report.qmd --to pdf
	@echo "Final report generated at docs/reports/final_report.pdf"

# ——— Notebooks ———
.PHONY: notebooks
notebooks:
	@echo "📓 Rendering & executing EDA notebooks…"
	@jupyter nbconvert --to notebook --execute notebooks/eda_conveyer_belt.ipynb --inplace
	@jupyter nbconvert --to notebook --execute notebooks/eda_high_temp_fan.ipynb --inplace
	@jupyter nbconvert --to notebook --execute notebooks/eda_tube_mill.ipynb --inplace

# ——— Archived scripts ———
.PHONY: archive-feature-eng-rar archive-model archive-rnn archive

archive-feature-eng-rar:
	@echo "🏗️  Running feature_eng_rar.py…"
	@python model/archive/feature_eng_rar.py

archive-model:
	@echo "🏗️  Running model.py…"
	@python model/archive/model.py

archive-rnn:
	@echo "🏗️  Running rnn_model.py…"
	@python model/archive/rnn_model.py

archive: archive-feature-eng archive-model archive-rnn
	@echo "✅ All archived model scripts complete!"