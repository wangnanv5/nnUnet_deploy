try:
    import cupyx.scipy.ndimage
    import cupy as cp
except ImportError as e:
    pass

import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from scipy.ndimage import zoom

class OnnxInfer():
    def __init__(self,classes_count,model_stream,use_gpu,plans_json_data):
        self.classes_count = classes_count
        self.model_stream = model_stream 
        self.use_gpu = use_gpu
        self.plans_json_data = plans_json_data
        self.gpu_available = False
        self.ort_session = self.load_model(model_stream,use_gpu)
        
    def load_model(self,model_stream,use_gpu):
        if not use_gpu:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 40  # 设置Intra-Op线程数（根据CPU核心数调整）
            session_options.inter_op_num_threads = 2
            
            session = ort.InferenceSession(model_stream.read(),session_options=session_options,providers=["CPUExecutionProvider"])
            print("use_gpu为False, 正在使用 CPU 推理.")
            return session
        try:
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError("无法使用 CUDA 推理引擎，请检查 CUDA 是否可用.")
            
            session = ort.InferenceSession(model_stream.read(),providers=["CUDAExecutionProvider"])
            print("ONNX Runtime GPU 版本可用，正在使用 GPU 推理")
            self.gpu_available = True
            return session
        
        except Exception as e:
            print("回退到 ONNX Runtime CPU 版本")
            session = ort.InferenceSession(model_stream.read(),providers=["CPUExecutionProvider"])
            return session

    def check_gpu_memory_and_allocate(self,sizes,dtype=cp.float32):
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()  
        used_mem = total_mem - free_mem

        free_mem_gb = free_mem / (1024**3)
        total_mem_gb = total_mem / (1024**3)
        used_mem_gb = used_mem / (1024**3)

        print(f"GPU 显存信息:")
        print(f"  总显存: {total_mem_gb:.1f} GB")
        print(f"  已用显存: {used_mem_gb:.1f} GB")
        print(f"  可用显存: {free_mem_gb:.1f} GB")

        required_gb = (cp.dtype(dtype).itemsize * sizes ) / (1024**3)

        print(f"所需显存: {required_gb:.1f} gb")
        if required_gb > free_mem_gb:
            print(f"显存不足，分块处理（chunking）方式处理\n")
            return True
        else:
            return False
        
    def run_infer_ONNX(self,input_array,slicers,gaussian_array,parameters_dict):
        if self.use_gpu and self.gpu_available:
            return self.run_infer_onnx_gpu(input_array,slicers,gaussian_array,parameters_dict)
        else:
            return self.run_infer_onnx_cpu(input_array,slicers,gaussian_array,parameters_dict)
    
    def run_infer_onnx_cpu(self,input_array,slicers,gaussian_array,parameters_dict):
        gaussian_array = np.asarray(gaussian_array[None], dtype=np.float32)
        
        predicted_logits = np.zeros(([self.classes_count] + list(input_array.shape)),dtype=np.float32)
        n_predictions = np.zeros((input_array.shape),dtype=np.float32)[None]
        
        patch_size = self.plans_json_data['configurations']['3d_fullres']['patch_size']
        output_shape = (1, self.classes_count) +  tuple(patch_size)
        ort_output = np.zeros(output_shape, dtype=np.float32)
        ort_output = np.ascontiguousarray(ort_output)        
        
        for slicer in tqdm(slicers):
            input_array_part = input_array[slicer][None][None]
            input_array_part = np.ascontiguousarray(input_array_part,dtype=np.float32)            
            ort_inputs = {self.ort_session.get_inputs()[0].name:input_array_part}
            ort_output = self.ort_session.run(None,ort_inputs)[0][0]            
            
            predicted_logits[:,slicer[0],slicer[1],slicer[2]] += ort_output * gaussian_array
            n_predictions[:,slicer[0],slicer[1],slicer[2]] += gaussian_array
        
        for i in range(self.classes_count):
            predicted_logits[i] = predicted_logits[i] / n_predictions

        del gaussian_array, n_predictions
        
        slicer_revert_padding = parameters_dict.pad_bbox
        predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding])]
        predicted_logits = np.argmax(predicted_logits,0)
        
        zoom_factors = [t / o for t, o in zip(parameters_dict.shape_after_crop, predicted_logits.shape)]
        predicted_logits = zoom(predicted_logits,zoom_factors,order=0)
        
        slicer = tuple([slice(*i) for i in parameters_dict.crop_bbox])
        segmentation_reverted_cropping = np.zeros(parameters_dict.origin_shape,dtype=np.uint16)
        segmentation_reverted_cropping[slicer] = predicted_logits
        image_data = segmentation_reverted_cropping.astype(np.uint8)
        return image_data
    
    def run_infer_onnx_gpu(self,input_array,slicers,gaussian_array,parameters_dict):
        gaussian_array = cp.asarray(gaussian_array[None], dtype=np.float32)
        
        predicted_logits = cp.zeros(([self.classes_count] + list(input_array.shape)),dtype=np.float32)
        n_predictions = cp.zeros((input_array.shape),dtype=np.float32)[None]
        
        patch_size = self.plans_json_data['configurations']['3d_fullres']['patch_size']
        output_shape = (1, self.classes_count) +  tuple(patch_size)
        ort_output = cp.zeros(output_shape, dtype=cp.float32)
        ort_output = cp.ascontiguousarray(ort_output)
        
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        io_binding = self.ort_session.io_binding()
        
        for slicer in tqdm(slicers):
            input_array_part = input_array[slicer][None][None]
            input_array_part = cp.array(input_array_part, dtype=cp.float32)
            input_array_part = cp.ascontiguousarray(input_array_part)            
            
            io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=cp.float32, shape=tuple(input_array_part.shape), buffer_ptr=input_array_part.data.ptr)
            io_binding.synchronize_inputs()
            io_binding.bind_output(name=output_name,device_type='cuda',device_id=0,element_type=np.float32,shape=output_shape,buffer_ptr=ort_output.data.ptr)
            
            self.ort_session.run_with_iobinding(io_binding) 
            ort_output_result = ort_output[0]
            
            predicted_logits[:,slicer[0],slicer[1],slicer[2]] += (ort_output_result * gaussian_array)
            n_predictions[:,slicer[0],slicer[1],slicer[2]] += gaussian_array
        
        for i in range(self.classes_count):
            predicted_logits[i] = predicted_logits[i] / n_predictions

        del gaussian_array, n_predictions
        
        slicer_revert_padding = parameters_dict.pad_bbox
        predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding])]
        predicted_logits = cp.argmax(predicted_logits,0)
        
        zoom_factors = [t / o for t, o in zip(parameters_dict.shape_after_crop, predicted_logits.shape)]
        predicted_logits = cupyx.scipy.ndimage.zoom(predicted_logits,zoom_factors,order=0)
        
        slicer = tuple([slice(*i) for i in parameters_dict.crop_bbox])
        segmentation_reverted_cropping = np.zeros(parameters_dict.origin_shape,dtype=np.uint16)
        segmentation_reverted_cropping[slicer] = predicted_logits.get()
        image_data = segmentation_reverted_cropping.astype(np.uint8)
        
        del predicted_logits,ort_output
        cp.get_default_memory_pool().free_all_blocks()
        
        return image_data