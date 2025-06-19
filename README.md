# GPS-Lung

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



### 5. Download Data





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
