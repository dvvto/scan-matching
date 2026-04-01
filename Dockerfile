FROM kirillmouraviev/prism-topomap:cuda12.1-ros-noetic-habitat_v0.2.3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV PATH="/home/docker_prism/.local/bin:${PATH}"
ENV PYTHONPATH="/opt/sonata:/opt/Swin3D:/opt/Swin3D_Task/SemanticSeg:${PYTHONPATH}"

# ---- Python base dependencies ----

RUN pip install --no-cache-dir \
    timm \
    addict \
    huggingface_hub \
    torchaudio==2.1.2 \
    openmim==0.3.3 \
    spconv-cu118 \
    torch-geometric \
    jupyter

RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# ---- OpenMMLab stack ----

RUN mim install mmcv-full \
 && mim install mmdet \
 && mim install mmsegmentation

# ---- OctFormer ----

RUN git clone --depth 1 https://github.com/octree-nn/octformer.git \
 && cd octformer \
 && pip install --no-cache-dir -r requirements.txt

# ---- DWConv CUDA extension ----

RUN git clone --depth 1 https://github.com/octree-nn/dwconv.git \
 && pip install ./dwconv

# ---- MMDetection3D ----

RUN git clone --depth 1 https://github.com/open-mmlab/mmdetection3d.git \
 && pip install -e ./mmdetection3d

# ---- Jupyter config ----

ENV JUPYTER_PASSWORD=jupyter
ENV JUPYTER_TOKEN=jupyter

# ---- Flash Attention (required by Sonata/PTv3) ----

RUN pip install --no-build-isolation flash-attn einops

# ---- Sonata (self-supervised PTv3, Facebook Research) ----

RUN git clone --depth 1 https://github.com/facebookresearch/sonata.git /opt/sonata \
 && cd /opt/sonata \
 && pip install --no-deps -e .

# ---- Swin3D (Microsoft) ----

RUN git clone --depth 1 https://github.com/microsoft/Swin3D.git /opt/Swin3D \
 && cd /opt/Swin3D \
 && pip install -r requirements.txt \
 && python setup.py build_ext --inplace \
 && python setup.py install

# ---- Swin3D_Task (semantic segmentation code) ----

RUN git clone --depth 1 https://github.com/Yukichiii/Swin3D_Task.git /opt/Swin3D_Task

# ---- gdown for downloading weights ----

RUN pip install --no-cache-dir gdown

# ---- Download Swin3D-S ScanNet fine-tuned weights ----

RUN mkdir -p /data/weights/swin3d_scannet \
 && gdown '1ttObwwJMrW2_9gd3xgrvpzddY0khIAbd' \
         -O /data/weights/swin3d_scannet/swin3d_s_scannet.pth

# ---- Sonata weights are downloaded on first use from HuggingFace ----
# (cached in ~/.cache/sonata/ckpt)

ENTRYPOINT ["/startup.sh"]
CMD ["/bin/bash"]
