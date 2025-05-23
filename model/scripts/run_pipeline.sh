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
    echo "Processing device: $device"
    if "$PYTHON_PATH" "$PROJECT_ROOT/src/preprocess.py" --device "$device" --aws; then
        echo "Successfully processed $device"
        return 0
    else
        echo "Error processing $device"
        return 1
    fi
}

devices=(
    "1#High-Temp Fan"
    "8#Belt Conveyer"
    "Tube Mill"
)

echo "=== Starting Data Preprocessing ==="
failed_devices=()
for device in "${devices[@]}"; do
    if ! run_preprocessing "$device"; then
        failed_devices+=("$device")
    fi
done

echo "=== Preprocessing Summary ==="
if [ ${#failed_devices[@]} -eq 0 ]; then
    echo "All devices processed successfully"
else
    echo "The following devices failed to process:"
    printf '%s\n' "${failed_devices[@]}"
    exit 1
fi

# TODO: Add model training/prediction steps here
# echo "=== Starting Model Training ==="
# echo "=== Starting Model Prediction ==="

echo "Pipeline execution completed at $(date)"
exit 0
