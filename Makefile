.PHONY: all download preprocess features train clean

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
