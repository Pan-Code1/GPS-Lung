import pydicom
import numpy as np
import os
import SimpleITK as sitk
import time
import h5py

from reader import DicomReader

class DicomWriter:
    def __init__(self, minHU=-1000,
                        maxHU=1000,
                        temp_path=None,
                        data_key=None):
        assert temp_path is not None
        self.HU = (minHU, maxHU)
        self.temp_path = temp_path
        self.data_key = data_key if data_key is not None else "synthesis_data"

    def load_temp(self, temp_file):
        f = h5py.File(os.path.join(self.temp_path, temp_file), "r")
        temp_data_dict = {}
        dicom_src = f["dicom_src"][()]
        data_arr = f[self.data_key][:]

        return dicom_src, data_arr

    def postprocess(self, data_arr):
        HU_range = self.HU[1] - self.HU[0]
        denorm_data = data_arr * HU_range - self.HU[0]
        postprocess_data = denorm_data.astype(np.int16)

        return postprocess_data

    def write(self, data_dict, save_path, name_type="SeriesID"):
        dicom_src, data_arr = self.load_temp(data_dict["temp_file"])
        data_arr = self.postprocess(data_arr)
        print(f"writing data from {dicom_src}")
        if name_type == "SeriesID":
            save_name = data_dict["dicom_info"].SeriesInstanceUID
        else:
            print("Unsupported name type")
        print(f"save data as {name_type} {save_name}")
        os.makedirs(os.path.join(save_path, save_name), exist_ok=True)
        dicom_info = data_dict["dicom_info"]
        for i, slice in enumerate(data_arr):
            ds = dicom_info.copy()
            slice = slice - 1000
            ds.PixelData = slice.tobytes()
            ds.InstanceNumber = str(i+1)

            ds.save_as(os.path.join(save_path, save_name, f"dicom_{i+1:0=3d}"))


# if __name__ == "__main__":
#     root_path = r"\\10.13.51.79\data\CT\CBCT\wuhanxiehe\paired\CBCT-100\CBCT-100\CBCT\AN_XIU_FANG_P5881566"
#     temp_path = r"D:\data\temp"
#     reader = DicomReader(minHU=-1000, maxHU=500, temp_path=temp_path) #输出缓存路径用来保存中间结果，不然dicom数据都放在内存会爆内存
#     dicom_dataset = reader.read(root_path)
#     print(f"dicom dataset ({len(dicom_dataset)})")
#
#     save_path = r"D:\data\temp\save"
#     writer = DicomWriter(minHU=-1000, maxHU=500, temp_path=temp_path, data_key="dicom_data")
#     for raw_data_dict in dicom_dataset:
#         writer.write(raw_data_dict, save_path)
