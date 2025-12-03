import os
import zipfile
import jwt
import time
# from jose import jwt, exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa
# from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import padding

def package_files(output_path, *file_paths):
    with zipfile.ZipFile(output_path, 'w') as zipf:
        for file in file_paths:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")
            zipf.write(file, arcname=os.path.basename(file))

def generate_iv(iv_path):
    iv = os.urandom(16)
    with open(iv_path, 'wb') as f:
        f.write(iv)

class JWTUtils():
    @staticmethod
    def generate_key_pair():
        private_key = rsa.generate_private_key(public_exponent=65537,key_size=2048)
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return private_key, public_key, private_pem, public_pem
    
    @staticmethod
    def generate_jwt_token(private_key, expiration_minutes=1):
        payload = {
            'iat': int(time.time()),
            'exp': int(time.time()) + expiration_minutes * 60
        }
        token = jwt.encode(payload, private_key, algorithm='RS256')
        return token
    
    @staticmethod
    def verify(token,  public_key):
        try:
            jwt.decode(token, public_key, algorithms=['RS256'])
        except jwt.ExpiredSignatureError:
            return "Token已经过期, 请重新申请."
        except jwt.InvalidTokenError:
            return "Token错误."
        except Exception as e:
            return f"未知错误: {str(e)}"

        return None
        
class OnnxCryptor:
    def __init__(self, key: bytes = b"this_is_a_32_byte_key_for_AES256", iv: bytes = None):
        if len(key) != 32:
            raise ValueError("密钥长度必须为 32 字节(AES-256)")
        
        # with open(iv_path, 'rb') as f:
        #     iv = f.read()
            
        if len(iv) != 16:
            raise ValueError("随机向量IV长度必须为 16 字节")
        
        self.key = key
        self.iv = iv
        self.cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())

    def encrypt_model_filepath(self, input_path: str, output_path: str):

        with open(input_path, 'rb') as f:
            raw_data = f.read()

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(raw_data) + padder.finalize()
        encryptor = self.cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        print(f"Encrypted length: {len(encrypted)}")  # 应为 16 的倍数

        with open(output_path, 'wb') as f:
            f.write(encrypted)
        print(f"模型加密完成：\n  原始大小: {len(raw_data)}\n  补齐后大小: {len(padded_data)}\n  密文保存至: {output_path}\n")

    def decrypt_model(self, encrypted):
            
        assert len(encrypted) % 16 == 0, f"加密数据长度不是 16 字节的倍数，当前为 {len(encrypted)}"

        decryptor = self.cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
def extract_private_key(filename="your_file.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()

    capture = False
    private_key_lines = []

    for line in lines:
        if line.startswith("private_key="):
            line = line[len("private_key="):]
            capture = True

        if capture:
            private_key_lines.append(line.strip())

        if "-----END PRIVATE KEY-----" in line:
            break 

    return '\n'.join(private_key_lines)

def extract_public_key(filename="your_file.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()

    capture = False
    private_key_lines = []

    for line in lines:
        if line.startswith("public_key="):
            line = line[len("public_key="):]
            capture = True

        if capture:
            private_key_lines.append(line.strip())

        if "-----END PUBLIC KEY-----" in line:
            break  

    return '\n'.join(private_key_lines)