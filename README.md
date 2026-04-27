# Dental AI Fusion Platform 🦷

A clinical-style proof-of-concept platform for multimodal dental imaging using MONAI, PyTorch, and Streamlit.

## Overview

This project demonstrates an AI-assisted dental imaging workflow that combines:

- Panoramic X-ray
- CBCT volume data
- Soft tissue / facial surface input

The platform provides fusion visualization, CBCT exploration, segmentation overlay, DICOM-ready ingestion, heatmap visualization, and downloadable clinical-style reports.

## Features

- Multimodal image upload
- Built-in synthetic demo data fallback
- CBCT zip size slack support
- DICOM-ready CBCT ingestion
- Axial, coronal, and sagittal CBCT viewer
- Brightness, contrast, and zoom controls
- AI fusion output visualization
- Simulated / fallback segmentation overlay
- AI attention heatmap
- ROI-style detected region display
- PDF clinical report export
- Interactive glossary and term guide
- Presentation mode for client demos

## Tech Stack

- Python
- Streamlit
- PyTorch
- MONAI 
- NumPy
- Matplotlib
- Pillow
- Pydicom
- ReportLab

## Project Structure

```text
monai_dental_project/
├── app.py
├── model.py
├── data.py
├── transforms.py
├── segmentation_model.py
├── demo_sections.py
├── requirements.txt
└── README.md