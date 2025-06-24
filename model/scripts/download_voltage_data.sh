#!/bin/bash
set -e

DATA_DIR="data/voltage"
ZIP_NAME="voltage.zip"
FILE_ID="10qpmMubE_BZK_Uwgi3K80DK-X2h6k88T"
TEMP_UNZIP_DIR="temp_unzip_voltage"

# Make sure the target folder exists
mkdir -p "$DATA_DIR"

# Skip if already populated
if [ -z "$(ls -A "$DATA_DIR")" ]; then
  echo "📥 Downloading $ZIP_NAME from Google Drive..."
  gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$ZIP_NAME"

  echo "📦 Unzipping to temporary folder..."
  rm -rf "$TEMP_UNZIP_DIR"
  unzip -q "$ZIP_NAME" -d "$TEMP_UNZIP_DIR"

  echo "📁 Moving contents to $DATA_DIR..."
  mv "$TEMP_UNZIP_DIR"/voltage/* "$DATA_DIR"

  echo "🧹 Cleaning up..."
  rm -rf "$TEMP_UNZIP_DIR" "$ZIP_NAME"

  echo "✅ Done. Files are in $DATA_DIR"
else
  echo "⏩ Skipped: voltage data already exists in $DATA_DIR"
fi
