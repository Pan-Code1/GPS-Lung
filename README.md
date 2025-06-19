# GPS-Lung

# CBCT-CT Conversion System

A deep learning-based CBCT to CT image conversion system using diffusion models for high-quality medical image modality transformation.

## Project Overview

This project implements the CBCT to CT image generation method from the GPS-Lung system, as described in the paper "Generative AI–Powered Navigation System for Precision Puncture: A Multicenter Randomized Controlled Trial Evaluating Efficacy and Safety". 

The system implements a CBCT (Cone Beam Computed Tomography) to CT image conversion system based on diffusion models. The system adopts a client-server architecture and supports batch processing of DICOM format medical image data.

### Related Publication

**Paper**: "Generative AI–Powered Navigation System for Precision Puncture: A Multicenter Randomized Controlled Trial Evaluating Efficacy and Safety"

**Method**: GPS-Lung CBCT to CT image generation implementation

This repository contains the practical implementation of the CBCT-to-CT conversion component used in the GPS-Lung navigation system for precision medical procedures.

## Guide

### 1. Clone Repository

```bash
git clone <repository-url>
cd CBCT_run
```

### 2. Create Virtual Environment

```bash
conda create -n cbct_env python=3.8
conda activate cbct_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model

https://drive.google.com/file/d/1f4GF29ZXa8x1nvGffWMY-PpyDDBxz60u/view?usp=drive_link


### 5. Download Data

https://drive.google.com/file/d/176WkRjppaPSZG_r6QPudvH3eH8yzlSTY/view?usp=drive_link



## Usage

### System Startup

The system adopts a client-server separation architecture, requiring separate startup of server and client.

#### 1. Start Server

In the first terminal window:

```bash
python server.py
```


#### 2. Start Client

In the second terminal window:

```bash
python client.py --input_path /path --output_path /path 
```

#### Parameter Description

- `--input_path`: Input DICOM data path
- `--output_path`: Output result save path
- `--flip_to_supine`: (Optional) Add this parameter if input images are prone position to convert to supine position
