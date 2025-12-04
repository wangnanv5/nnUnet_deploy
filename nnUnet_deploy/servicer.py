import os
import sys
import zipfile
import json
import shutil
import tempfile
import queue
import threading
import cupy as cp
import numpy as np
import SimpleITK as sitk
from io import BytesIO
from pathlib import Path

from nnUnet_deploy.munch import DefaultMunch
from nnUnet_deploy.infer import OnnxInfer
from nnUnet_deploy.preprocess import preprocess
from nnUnet_deploy.utils import OnnxCryptor

class Servicer():
    def __init__(self,config:dict):
        self.config = config
        
        self.check_path()
        self.load_from_package(self.config.model_path)
        self.onnx_infer = OnnxInfer(self.classes_count,self.model_stream,self.config.use_gpu,self.plans_json_data)
    
    def save_nifti_safe(self,image, chinese_path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = os.path.join(tmp_dir, "temp_image.nii.gz")
            sitk.WriteImage(image, temp_path)
            shutil.copy2(temp_path, chinese_path)
            print(f"成功保存到: {chinese_path}")
        return chinese_path
    
    def check_path(self):
        if not Path(self.config.model_path).exists():
            sys.exit('Model path is not exists, please check it.')
            
        self.input_path = Path(self.config.input_path)
        
        if not self.input_path.exists():
            sys.exit('Input path is not exists, please check it.')
        
        if self.input_path.is_dir() and self.config.file_suffix != 'dcm':
            self.infer_file_list = sorted(self.input_path.glob('*'+self.config.file_suffix))
            if len(self.infer_file_list) == 0:
                sys.exit('Input folder have no such files, please check it.')
        elif self.input_path.is_dir() and self.config.file_suffix == 'dcm':
            self.infer_file_list = [self.input_path]
        else:
            self.infer_file_list = [self.input_path]
            
        print(f'Current infer {len(self.infer_file_list)} files.')
        self.output_path = Path(self.config.output_path)
        
        if not self.output_path.exists():
            print(f'Output path does not exist,create it.')
            self.output_path.mkdir(parents=True)
    
    def load_from_package(self,pkg_path):
        with zipfile.ZipFile(pkg_path) as zipf:
            file_list = zipf.namelist()
            
            iv_files =  next(f for f in file_list if f.endswith('.iv'))
            with zipf.open(iv_files) as f:
                iv_data = f.read()    
            
            encry_files =  next(f for f in file_list if f.endswith('.encry'))
            with zipf.open(encry_files) as f:
                model_data = f.read()
            
            onnx_cryptor = OnnxCryptor(iv=iv_data)
            model_bytes = onnx_cryptor.decrypt_model(model_data)
            self.model_stream = BytesIO(model_bytes)
            
            with zipf.open('dataset.json') as f:
                self.dataset_json_data = DefaultMunch.fromDict(json.load(f))
                self.classes_count = len(self.dataset_json_data.labels)
                
            with zipf.open('plans.json') as f:
                self.plans_json_data = DefaultMunch.fromDict(json.load(f))
    
    def load_and_preprocess(self,image_path):
        output_file_path = self.output_path / image_path.name
        if output_file_path.exists() and not self.config.overwrite:
            return None
        
        print(f'Load {image_path}')
        
        ct_image = sitk.ReadImage(image_path)
        ct_array = sitk.GetArrayFromImage(ct_image)
        spacing = ct_image.GetSpacing()[::-1]
        
        result = []
        if self.config.use_chunk == True:
            z_size = ct_array.shape[0]
            data_queue = [ct_array[:int(z_size / 2)],ct_array[int(z_size / 2):]]
            for data in data_queue:
                preprocessed_array,slicers,gaussian,parameters_dict = preprocess(data, spacing,self.plans_json_data)
                result.append([preprocessed_array,slicers,gaussian,parameters_dict])
        else:
            preprocessed_array_one,slicers,gaussian,parameters_dict = preprocess(ct_array, spacing,self.plans_json_data)
            result.append([preprocessed_array_one,slicers,gaussian,parameters_dict])

        return result,ct_image,image_path.name

    def load_queue(self,file_paths,data_queue):
        for path in file_paths:
            data = self.load_and_preprocess(path)
            if data == None:
                continue
            data_queue.put(data)
        data_queue.put(None)     
        
    def _run_infer(self):
        # 多线程加载推理 目前容易造成程序假死 不好用
        data_queue = queue.Queue(maxsize=10)
        loader = threading.Thread(target=self.load_queue, args=(self.infer_file_list, data_queue),daemon=True)
        loader.start()
        
        while True:
            data,ct_image,file_name = data_queue.get()
            if data is None:
                break
            
            if self.config.use_chunk == True:
                result = []
                for data_one in data:
                    result.append(self.onnx_infer.run_infer_ONNX(data_one[0],data_one[1],data_one[2],data_one[3]))
                ct_array = np.concatenate(result)
            else:
                ct_array = self.onnx_infer.run_infer_ONNX(data[0][0],data[0][1],data[0][2],data[0][3])
            
            image = sitk.GetImageFromArray(ct_array)
            image.CopyInformation(ct_image)
            sitk.WriteImage(image,self.output_path / file_name)
    
    def run_infer(self):
        for index in range(len(self.infer_file_list)): 
            if self.config.file_suffix == 'dcm' and self.input_path.is_dir():
                output_file_path = self.output_path / (self.infer_file_list[index].name + '.nii.gz')
            elif self.config.file_suffix == 'nii' or self.config.file_suffix == 'nii.gz':
                output_file_path = self.output_path / self.infer_file_list[index].name
                
            if output_file_path.exists() and not self.config.overwrite:
                continue
            
            print(f'Run {index + 1} / {len(self.infer_file_list)} files, current file is {self.infer_file_list[index]}')
            
            if self.config.file_suffix == 'dcm' and self.input_path.is_dir():
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(self.input_path)
                reader.SetFileNames(dicom_names)
                ct_image = reader.Execute()
                ct_array = sitk.GetArrayFromImage(ct_image)     
                spacing = ct_image.GetSpacing()[::-1]
            elif self.config.file_suffix == 'nii' or self.config.file_suffix == 'nii.gz':
                ct_image = sitk.ReadImage(self.infer_file_list[index])
                ct_array = sitk.GetArrayFromImage(ct_image)     
                spacing = ct_image.GetSpacing()[::-1]
            else:
                print('please check input file suffix')
                                                                        
            # all_bone_path = output_file_path.parent.parent / 'all_bone' /  self.infer_file_list[index].name
            # all_bone_image = sitk.ReadImage(all_bone_path)
            # all_bone_array = sitk.GetArrayFromImage(all_bone_image)
            # all_bone_array[all_bone_array != 6] = 0
            
            # ct_array[np.isnan(ct_array) | np.isinf(ct_array)] = 0
            # nonzero_indices = np.nonzero(all_bone_array)
            # min_coords = np.min(nonzero_indices[0])
            # max_coords = np.max(nonzero_indices[0])
            # ct_array[:min_coords - 20] = 0
            # ct_array[max_coords + 20:] = 0
            
            # min_coords = np.min(nonzero_indices[1])
            # max_coords = np.max(nonzero_indices[1])
            # ct_array[:,:min_coords - 20] = 0
            # ct_array[:,max_coords + 20:] = 0
            
            # min_coords = np.min(nonzero_indices[2])
            # max_coords = np.max(nonzero_indices[2])
            # ct_array[:,:,:min_coords - 20] = 0
            # ct_array[:,:,max_coords + 20:] = 0
            
            # # 下肢
            # all_bone_array[all_bone_array != 4] = 0
            # nonzero_indices = np.nonzero(all_bone_array)
            # min_coords = np.min(nonzero_indices[0])
            # ct_array[:min_coords - 10] = 0   
            
            #肋骨
            # all_bone_array[all_bone_array != 3] = 0
            # nonzero_indices = np.nonzero(all_bone_array)
            # min_coords = np.min(nonzero_indices[0])
            # max_coords = np.max(nonzero_indices[0])
            # ct_array[:min_coords - 20] = 0
            # ct_array[max_coords + 20:] = 0
            
            # min_coords = np.min(nonzero_indices[1])
            # max_coords = np.max(nonzero_indices[1])
            # ct_array[:,:min_coords - 20] = 0
            # ct_array[:,max_coords + 20:] = 0
            
            # min_coords = np.min(nonzero_indices[2])
            # max_coords = np.max(nonzero_indices[2])
            # ct_array[:,:,:min_coords - 20] = 0
            # ct_array[:,:,max_coords + 20:] = 0            
            
            if self.config.use_chunk == True:
                result = []
                chunk_count = self.config.chunk_count
                data_queue = np.array_split(ct_array, chunk_count, axis=0)
                
                for data in data_queue:
                    preprocessed_array,slicers,gaussian,parameters_dict = preprocess(data, spacing,self.plans_json_data)
                    cp.get_default_memory_pool().free_all_blocks()
                    result.append(self.onnx_infer.run_infer_ONNX(preprocessed_array,slicers,gaussian,parameters_dict))
                    del preprocessed_array
                    cp.get_default_memory_pool().free_all_blocks()
                    
                ct_array = np.concatenate(result)
            else:
                ct_array,slicers,gaussian,parameters_dict = preprocess(ct_array, spacing,self.plans_json_data)
                ct_array = self.onnx_infer.run_infer_ONNX(ct_array,slicers,gaussian,parameters_dict)
                
            new_image = sitk.GetImageFromArray(ct_array)
            new_image.CopyInformation(ct_image)
            self.save_nifti_safe(new_image, output_file_path)

