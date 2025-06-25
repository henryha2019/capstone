#!/bin/bash
set -e

DATA_DIR="data/voltage"
ZIP_NAME="voltage.zip"
FILE_ID="1dsNqNYfvnSS6W5rAwDHOULK0m53Artsp"
TEMP_UNZIP_DIR="temp_unzip_voltage"

# Make sure the target folder exists
mkdir -p "$DATA_DIR"

# Skip if already populated
if [ -z "$(ls -A "$DATA_DIR")" ]; then
  echo "📥 Downloading $ZIP_NAME from Google Drive (large file workaround)..."
  gdown "https://drive.google.com/uc?export=download&id=${FILE_ID}&confirm=t" -O "$ZIP_NAME"

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