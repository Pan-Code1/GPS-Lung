import sys
import os
import argparse

import requests
import json

from config import Config
from reader import DicomReader
from writer import DicomWriter

cfg = Config()

def generate(raw_data_dict, flip):
    temp_file = raw_data_dict["temp_file"]
    url = f'http://{cfg.host}:{cfg.port}{cfg.server_url}'
    print(f"Waiting server response: {url}")
    msg = {"temp_file": temp_file, "flip": flip}
    msg = json.dumps(msg)
    response = requests.post(url, data=msg)
    if response.status_code != 200:
        print("request failed: {response.status_code}")

    if response.text == "generate done":
        return 0
    else: 
        return -1


def run(input_path, output_path, flip):
    print("-"*50)
    print("Reading dicom data...")
    reader = DicomReader(minHU=cfg.minHU, maxHU=cfg.maxHU, temp_path=cfg.temp_path)
    dicom_dataset = reader.read(input_path)
    print("-"*50)
    for raw_data_dict in dicom_dataset:
        print("Generating images...")
        if generate(raw_data_dict, flip) < 0:
            print("Generate failed!")
            continue
        print("-"*50)
        print("Saving results...")
        writer = DicomWriter(minHU=cfg.minHU, maxHU=cfg.maxHU, temp_path=cfg.temp_path)
        writer.write(raw_data_dict, output_path)
        print("-" * 50)
        print("Clean temp file")
        os.remove(os.path.join(cfg.temp_path, raw_data_dict["temp_file"]))
    for file in os.listdir(cfg.temp_path):
        if file[-4:] == ".tmp":
            pass
            os.remove(os.path.join(cfg.temp_path, file))

    print("-" * 50)

def close_server():
    url = f'http://{cfg.host}:{cfg.port}/shutdown'
    requests.post(url, data="shutdown")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Input path')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--flip_to_supine', action='store_true', help='Flip to supine')
    
    args = parser.parse_args()
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Flip to supine: {args.flip_to_supine}")

    input_path = args.input_path
    output_path = args.output_path
    flip_to_supine = args.flip_to_supine

    assert os.path.exists(input_path)
    if not os.path.exists(output_path):
        print(f"Create save path: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    run(input_path, output_path, flip_to_supine)

if __name__ == "__main__":
    main()