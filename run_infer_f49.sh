#!/bin/bash

# 检查 Conda 是否已初始化
if ! command -v conda &> /dev/null; then
    echo "Conda 未安装或未正确初始化，请先运行 'conda init bash' 并重新加载 Shell。"
    exit 1
fi

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保脚本中支持 conda activate
conda activate MGDIT

# 检查环境是否激活成功
if [[ $? -ne 0 ]]; then
    echo "Conda 环境 'mdD' 激活失败，请确保该环境存在。"
    exit 1
fi


# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 执行推理命令
# python ./scripts/inference_magicdrive.py configs/magicdrive/inference/fullx224x400_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
#     --cfg-options model.from_pretrained=./ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt \
#     force_pad_h_for_sp_size=4 \
#     num_frames=17 cpu_offload=true \
#     force_image=true \
#     scheduler.type=rflow-slice


torchrun --standalone --nproc_per_node 1 \
    scripts/inference_magicdrive.py \
    configs/magicdrive/inference/fullx224x400_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
    --cfg-options model.from_pretrained=./ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt \
    force_pad_h_for_sp_size=4 \
    num_frames=49 cpu_offload=true \
    force_image=true \
    scheduler.type=rflow-slice