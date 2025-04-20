import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from segment_anything import sam_model_registry
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import urllib.request
import zipfile
import glob

# FFT 기반 셀프어텐션 클래스 정의
class FFTSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 쿼리, 키, 밸류 프로젝션 레이어
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 쿼리, 키, 밸류 프로젝션
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # FFT 기반 어텐션 계산
        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        
        # 복소수 곱셈 (컨볼루션에 해당)
        res = q_fft * k_fft.conj()
        
        # 역변환
        attn_output = torch.fft.irfft(res, dim=-1, n=self.head_dim)
        
        # 결과와 밸류의 곱
        output = attn_output * v
        
        # 원래 형태로 변환
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output

# SAM 모델 수정하는 함수
def replace_attention_with_fft(model):
    """SAM 모델의 셀프어텐션 레이어를 FFT 버전으로 교체"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # MultiheadAttention의 embed_dim 가져오기
            embed_dim = module.embed_dim
            # FFT 어텐션으로 교체
            setattr(model, name, FFTSelfAttention(dim=embed_dim, num_heads=module.num_heads, dropout=module.dropout))
        else:
            replace_attention_with_fft(module)
    
    return model

# SAM 모델 다운로드 함수
def download_sam_model(model_type="vit_b", save_dir="./sam_checkpoints"):
    """SAM 모델 체크포인트를 다운로드하는 함수"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_path = os.path.join(save_dir, f"sam_{model_type}.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"SAM {model_type} 모델 다운로드 중...")
        urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
    
    return checkpoint_path

# 네일 세그멘테이션 데이터셋 클래스
class NailSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(1024, 1024)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        # 이미지와 마스크 경로 설정
        self.images_dir = os.path.join(data_dir, "trainset_nails_segmentation")
        self.labels_dir = os.path.join(self.images_dir, "labels")
        
        print(f"이미지 디렉토리: {self.images_dir}")
        print(f"마스크 디렉토리: {self.labels_dir}")
        
        # 모든 이미지 파일 찾기
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for file_path in glob.glob(os.path.join(self.images_dir, ext)):
                # labels 폴더 내부의 파일은 제외
                if 'labels' not in file_path:
                    base_name = os.path.basename(file_path)
                    mask_path = os.path.join(self.labels_dir, base_name)
                    if os.path.exists(mask_path):
                        self.image_files.append((file_path, mask_path))
        
        print(f"총 {len(self.image_files)}개의 이미지-마스크 쌍을 찾았습니다.")
        for i in range(min(5, len(self.image_files))):
            print(f"파일 샘플 {i+1}: {self.image_files[i]}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_files[idx]
        
        # 이미지와 마스크 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 그레이스케일로 로드
        
        # 이미지와 마스크 리사이즈
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        
        # 마스크를 이진 마스크로 변환 (임계값 127)
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8)  # 이진 마스크로 변환
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # 마스크를 텐서로 변환
        mask = torch.from_numpy(mask_np).long()
        
        return image, mask