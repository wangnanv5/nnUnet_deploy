import sys
import argparse
from pathlib import Path

from nnUnet_deploy.utils import JWTUtils
from nnUnet_deploy.servicer import Servicer

def get_parser():
    parser = argparse.ArgumentParser(description="AutoOrgan commercial closed-source version.")

    # parser.add_argument("-i",'--input_path',required=True, type=str, help="输入路径,可以是单个文件路径或者一个文件夹路径.")
    # parser.add_argument('-o',"--output_path",required=True, type=str,  help="输出结果路径,如果不存在则创建.")
    # parser.add_argument('-m',"--model_path", required=True, type=str, help="pkg格式的模型路径.")    
    # parser.add_argument('-fs',"--file_suffix", type=str,default="dcm", help="输入文件后缀,默认为nii.gz,目前支持nii/nii.gz/dcm.")    
    # parser.add_argument('-g',"--use_gpu", action="store_true",default=True, help="是否使用GPU进行加速,默认使用")    
    # parser.add_argument('-l',"--license",default='licence.txt', help="许可证文件.")
    # parser.add_argument('-ow',"--overwrite", type=bool, default=False, help="是否覆盖已存在的推理结果,默认为否.") 
    # parser.add_argument('-uc',"--use_chunk", type=bool, default=False, help="是否进行分块推理,默认为否.")  
    # parser.add_argument('-ucc',"--chunk_count", type=float, default=2, help="分块大小，仅当uc = True有效.")
    # parser.add_argument('-gid',"--gpu_id",type=str,default='0', help="The GPU used.")

    # for test
    parser.add_argument("-i",'--input_path', type=str,default='/home/wangnannan/workdir/ct/', help="")
    parser.add_argument('-o',"--output_path", type=str, required=False,default='/home/wangnannan/workdir/ct_segment/', help="Output path")
    parser.add_argument('-m',"--model_path", type=str, default='/home/wangnannan/workdir/AutoOrganCommercial/model/all_bone.pkg',help="The name of the model used")    
    parser.add_argument('-fs',"--file_suffix", type=str,default="nii.gz", help="You want the suffix of the inference files")    
    parser.add_argument('-ow',"--overwrite", type=bool, default=False, help="overwrite or not") 
    parser.add_argument('-uc',"--use_chunk", type=bool, default=False, help="chunk infer") 
    parser.add_argument('-g',"--use_gpu", action="store_true",default=True, help="Whether to use a GPU for acceleration")
    parser.add_argument('-l',"--license",default='/home/wangnannan/workdir/AutoOrganCommercial/licence.txt', help="License code.")
    parser.add_argument('-gid',"--gpu_id",type=str,default='0', help="The GPU used.")
    parser.add_argument('-ucc',"--chunk_count", type=float, default=4, help="分块大小，仅当uc = True有效.")
    
    args = parser.parse_args()
    return args

def verify_license(license_path : Path) -> bool:
    public_pem = """-----BEGIN PUBLIC KEY-----
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4MTVlyZ47mlywsb/Tu08
        QK32tkZZR+SvPDHXtDyKk6FqHfNmY0xSQsjEa0zZ34LWJ8SkTMRP9cs+e06CHbC/
        u7JI3XboakpBC04cJuywvSFkuEUfVfKF2pumvp5JT7o6pCjlJTXDUBHAhwosncMp
        9gujZWqOV7884Ch39140NjI/hnoE/V2OHsHjo6yG/q0zIEgWgmgA31rXSwLh6u7u
        LoGWXkDYpRv6n2dHemkRHbCRaheehqQEm2UezRzYb40+XeQ4k33Mlp1lmH4iP4lL
        kQpOX/S8eQggqTCeufgZGvMj/i15KML+a6AASfzR+gdUOaxBQ50pHKUPcwfChRwb
        xwIDAQAB
        -----END PUBLIC KEY-----"""
    
    if not license_path.exists():
        print(f"许可证文件不存在,请检查路径是否正确.")
        sys.exit(1)
    else:
        with open(license_path,"r") as f:
            license = f.read()
        result = JWTUtils.verify(license, public_pem)
        
        if result is None:
            print(f"许可证读取成功,正在运行推理,请稍等. ")
            return True
        else:
            print("许可证已过期,请重新申请")
            sys.exit(1)

def main_entry():
    args = get_parser()
    Servicer(args).run_infer()
        
if __name__ == '__main__':
    main_entry()