import pydicom
import numpy as np
import os
import SimpleITK as sitk
import time
import h5py

class DicomReader:
    def __init__(self, minHU=-1000,
                        maxHU=1000,
                        norm="MinMax",
                        temp_path=None):
        assert temp_path is not None
        self.temp_path = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.HU = (minHU, maxHU)
        if norm == "MinMax":
            self.norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        elif norm == "MeanStd":
            self.norm = lambda x: (x - np.mean(x)) / np.std(x)
        self.check_dicom_num = 10



    def get_dicom_dirs(self, root_path):
        dicom_path_list = []
        for root, dirs, files in os.walk(root_path):
            if dirs == []:
                dicom_path_list.append(root)

        return dicom_path_list

    def read_dicom(self, dicom_path):
        raw_data_list = []
        ds_list = []
        series_reader = sitk.ImageSeriesReader()
        for series_id in series_reader.GetGDCMSeriesIDs(dicom_path):
            print(f"reading dicom {dicom_path}\nSeriesID: {series_id}")
            try:
                dicoms = series_reader.GetGDCMSeriesFileNames(dicom_path, series_id)
                ds = pydicom.read_file(dicoms[0], force=True)
                series_reader.SetFileNames(dicoms)
                itk_img = series_reader.Execute()
                raw_data = sitk.GetArrayFromImage(itk_img).astype(np.float32)
                if raw_data.shape[0] < self.check_dicom_num:
                    print(f"Too few slices, please check:\n{dicom_path}, slice num({len(dicoms)})")
                    continue
                raw_data_list.append(raw_data)
                ds_list.append(ds)
            except Exception as e:
                print(e)

        if len(raw_data_list) > 1:
            print(f"Folder contains more than one Series, please check: \n{dicom_path}")

        return raw_data_list, ds_list

    def preprocess(self, raw_data):
        processed_data = np.clip(raw_data, self.HU[0], self.HU[1])
        processed_data = self.norm(processed_data)

        return processed_data

    def save_temp(self, data_dict, file_name: str):
        f = h5py.File(os.path.join(self.temp_path, file_name), "w")
        for k, v in data_dict.items():
            if isinstance(v, str):
                f.create_dataset(k, data=v, dtype=h5py.special_dtype(vlen=str))
            elif isinstance(v, np.ndarray):
                f.create_dataset(k, data=v)
            else:
                print(f"Unacceptable dtype: {k}, {type(v)}")

        f.close()

        return 0




    def read(self, root_path):
        dicom_path_list = self.get_dicom_dirs(root_path)

        dicom_dataset = []
        for dicom_path in dicom_path_list:
            raw_data_list, ds_list = self.read_dicom(dicom_path)
            if len(raw_data_list) == 0:
                continue
            else:
                raw_data = raw_data_list[0]
                ds_info = ds_list[0]
            processed_data = self.preprocess(raw_data)
            temp_file = "." + str(time.time()) + ".tmp"
            temp_data_dict, raw_data_dict = {}, {}
            temp_data_dict["dicom_src"] = dicom_path
            temp_data_dict["dicom_data"] = processed_data

            raw_data_dict["dicom_info"] = ds_info
            raw_data_dict["temp_file"] = temp_file
            self.save_temp(temp_data_dict, temp_file)
            dicom_dataset.append(raw_data_dict)
        return dicom_dataset

