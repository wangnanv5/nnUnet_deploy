from pathlib import Path

def test_generate_iv(iv_path):
    from AutoOrgan.utils import generate_iv
    generate_iv(iv_path)

def test_OnnxCryptor(model_path, encrypted_model_path,iv_path):
    from AutoOrgan.utils import OnnxCryptor
    
    with open(iv_path, 'rb') as f:
            iv = f.read()
    onnx_cryptor = OnnxCryptor(iv=iv)
    
    input_model_path = Path(model_path)
    encrypted_model_path = Path(encrypted_model_path)
    onnx_cryptor.encrypt_model_filepath(input_model_path, encrypted_model_path)
    
    # model_bytes = onnx_cryptor.decrypt_model(encrypted_model_path)
    # model_stream = BytesIO(model_bytes)

def test_package_files(output_path, *file_paths):
    from AutoOrgan.utils import package_files
    package_files(output_path, *file_paths)

def test_JWTUtils_once():
    from AutoOrgan.utils import JWTUtils
    private_key, public_key, private_pem, public_pem = JWTUtils.generate_key_pair()
    private_pem = private_pem.decode("utf-8")
    public_pem = public_pem.decode("utf-8")
    with open("keys.txt", "w",encoding="utf-8") as f:
        f.write(f"public_key={public_pem.strip()}\n")
        f.write(f"private_key={private_pem.strip()}\n")

def test_generate_jwt_token(expiration_minutes):
    from AutoOrgan.utils import JWTUtils,extract_private_key
    private_pem = extract_private_key("keys.txt")
    token = JWTUtils.generate_jwt_token(private_pem, expiration_minutes=expiration_minutes)
    return token

def test_verify(token):
    from AutoOrgan.utils import JWTUtils,extract_public_key
    # public_pem = extract_public_key("keys.txt")
    public_pem = """-----BEGIN PUBLIC KEY-----
                MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4MTVlyZ47mlywsb/Tu08
                QK32tkZZR+SvPDHXtDyKk6FqHfNmY0xSQsjEa0zZ34LWJ8SkTMRP9cs+e06CHbC/
                u7JI3XboakpBC04cJuywvSFkuEUfVfKF2pumvp5JT7o6pCjlJTXDUBHAhwosncMp
                9gujZWqOV7884Ch39140NjI/hnoE/V2OHsHjo6yG/q0zIEgWgmgA31rXSwLh6u7u
                LoGWXkDYpRv6n2dHemkRHbCRaheehqQEm2UezRzYb40+XeQ4k33Mlp1lmH4iP4lL
                kQpOX/S8eQggqTCeufgZGvMj/i15KML+a6AASfzR+gdUOaxBQ50pHKUPcwfChRwb
                xwIDAQAB
                -----END PUBLIC KEY-----"""
    
    result = JWTUtils.verify(token, public_pem)
    
    print(f"解加密结果: {result}")
    # print("\n等待后尝试解密...")
    # time.sleep(1)
    # result = JWTUtils.verify(token, public_pem)
    # print(f"过期后解密结果: {result}")
    
if __name__ == '__main__':
    
    # test_generate_iv('lower.iv')
    
    # test_OnnxCryptor('/home/wangnannan/workdir/AutoOrganCommercial/lower.onnx',
    #                  '/home/wangnannan/workdir/AutoOrganCommercial/lower.encry',
    #                  '/home/wangnannan/workdir/AutoOrganCommercial/lower.iv')
    
    # test_package_files('/home/wangnannan/workdir/AutoOrganCommercial/lower.pkg',
    #                    '/home/wangnannan/workdir/AutoOrganCommercial/lower.encry',
    #                    '/home/wangnannan/nnunet_dir/nnUNet_results/Dataset112_Lower/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json',
    #                    '/home/wangnannan/nnunet_dir/nnUNet_results/Dataset112_Lower/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json',
    #                    '/home/wangnannan/workdir/AutoOrganCommercial/lower.iv')
    # ### test_JWTUtils_once()
    
    token = test_generate_jwt_token(129600)
    with open("licence.txt", "w",encoding="utf-8") as f:
        f.write(token)
    # import time
    # time.sleep(5)
    # test_verify(token)

