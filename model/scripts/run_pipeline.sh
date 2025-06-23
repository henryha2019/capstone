#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_PATH=$(which python)
if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python not found in PATH"
    exit 1
fi
echo "Using Python from: $PYTHON_PATH"

echo "Starting pipeline execution at $(date)"

run_preprocessing() {
    local device="$1"
    echo "=== Step 1: Preprocessing device: $device ==="
    if "$PYTHON_PATH" "$PROJECT_ROOT/src/preprocess.py" --device "$device" --aws; then
        echo "Successfully preprocessed $device"
        return 0
    else
        echo "Error preprocessing $device"
        return 1
    fi
}

run_feature_engineering() {
    local device="$1"
    echo "=== Step 2: Feature Engineering for device: $device ==="
    if "$PYTHON_PATH" "$PROJECT_ROOT/src/feature_engineer.py" --device "$device" --aws; then
        echo "Successfully completed feature engineering for $device"
        return 0
    else
        echo "Error in feature engineering for $device"
        return 1
    fi
}

run_model_training() {
    local device="$1"
    echo "=== Step 3: Model Training for device: $device ==="
    
    # Define all models to train
    models=(
        "Baseline"
        "Ridge"
        "PolyRidgeDegree2"
        "RandomForest"
        "XGBoost"
        "SVR"
        "RuleTree"
    )
    
    failed_models=()
    for model in "${models[@]}"; do
        echo "--- Training model: $model ---"
        if "$PYTHON_PATH" "$PROJECT_ROOT/src/model.py" --model "$model" --device "$device" --aws --tune; then
            echo "Successfully completed training for model: $model"
        else
            echo "Error training model: $model"
            failed_models+=("$model")
        fi
    done
    
    # Check if any models failed
    if [ ${#failed_models[@]} -eq 0 ]; then
        echo "Successfully completed model training for all models on $device"
        return 0
    else
        echo "The following models failed to train:"
        printf '%s\n' "${failed_models[@]}"
        return 1
    fi
}

# Define all devices for preprocessing
devices=(
    "1#High-Temp Fan"
    "8#Belt Conveyer"
    "Tube Mill"
)

echo "=== Starting Data Preprocessing for All Devices ==="
failed_devices=()
for device in "${devices[@]}"; do
    if ! run_preprocessing "$device"; then
        failed_devices+=("$device")
    fi
done

echo "=== Preprocessing Summary ==="
if [ ${#failed_devices[@]} -eq 0 ]; then
    echo "All devices preprocessed successfully"
else
    echo "The following devices failed to preprocess:"
    printf '%s\n' "${failed_devices[@]}"
    exit 1
fi

# Only continue with feature engineering and model training for 8#Belt Conveyer
target_device="8#Belt Conveyer"

echo "=== Continuing Pipeline for $target_device ==="

# Run feature engineering
if ! run_feature_engineering "$target_device"; then
    echo "Pipeline failed at feature engineering step"
    exit 1
fi

# Run model training
if ! run_model_training "$target_device"; then
    echo "Pipeline failed at model training step"
    exit 1
fi

echo "=== Pipeline Summary ==="
echo "Preprocessing completed for all devices"
echo "Feature engineering and model training completed successfully for $target_device"

echo "Pipeline execution completed at $(date)"
exit 0
