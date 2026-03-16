# MusicSeg 边界检测当前上下文摘要

## 训练目标
**对标论文**: SongFormer (arXiv:2510.02797v2)  
**目标指标** (SongFormer HX 在 SongFormBench-HarmonixSet):
| 指标 | 论文值 | 当前最佳 | 差距 |
|------|--------|----------|------|
| ACC | 0.795 | ~0.64 | -0.15 |
| HR.5F | **0.703** | ~0.455 | -0.248 |
| HR3F | 0.784 | ~0.651 | -0.133 |

**训练环境**: GPU 机器（远程），本地做代码改动后同步。

## 关键发现

### 1. 多分辨率模型效果不佳
- multi_res 模型 HR.5F=0.17，低于 ms_transformer_v2 的 0.23
- 原因：整首歌训练导致 batch_size=1，训练不稳定
- 结论：**继续使用 ms_transformer_v2**

### 2. 分类器 focal loss 效果差
- focal loss: acc=0.21，远低于 CE+class_weight 的 0.61
- 结论：**分类器使用 SoftmaxFocalLoss (SOTA) 配合 EMA**

### 3. 分类器保存问题已修复
- best_state 初始化为 None 导致首次保存失败
- 已修复为初始化为当前模型状态

### 4. 最新指标（v25_final -> v26）
- **边界检测**: 
    - 引入 **Deep Supervision** (深层监督)，强制多尺度特征独立学习。
    - 增强 **Cross-Scale Skip Connections**。
    - 保持 Bi-directional Fusion 和 Contrastive Loss (weight=0.1)。
    - **v25_final 最佳**: HR.5F=39.4%, HR3F=63.4% (Epoch 101)。
    - **v26 计划**: 引入 **Gaussian Label Smoothing** (SOTA 核心Trick), **TVLoss1D**, **EMA**, **Dynamic Hybrid Loss** (Ramp-up)。
- **分类器**: 
    - **Transformer Encoder 升级至 4 层**，提升模型容量。
    - 保持 Sinusoidal PE 和 AMP 混合精度。
    - **v25_final 最佳**: Train ACC=89.9%, Val ACC=61.9%。严重的过拟合。
    - **v26 计划**: 
        - 引入 **SongFormer SOTA 架构**: `TimeDownsample1D` (kernel=5, stride=5) + `SEBlock` + `LayerNorm`。
        - 移除破坏性的 `MaxPool2d`，改用 `Conv2d` + `TimeDownsample1D`。
        - 引入 **EMA** (Exponential Moving Average) 稳定训练。
        - 引入 **SoftmaxFocalLoss** (SOTA)。
        - 优化保存策略：加权评分 (0.4*Train + 0.6*Val)。

## 论文关键技术（v26 集成）
1. **Deep Supervision**: 多尺度辅助损失 (SOTA) + **Aux Loss Decay** (动态衰减)
2. **Hybrid loss**: BCE + Focal 0.5/0.5 稳定混合 + **Ramp-up** (渐进式)
3. **Gaussian Label Smoothing**: 边界标签高斯平滑 (Sigma=3.33)
4. **TVLoss1D**: 边界平滑度正则化
5. **EMA**: 指数移动平均 (Decay=0.999)
6. **Structure Contrastive Loss**: 结构对比损失 (weight=0.1)
7. **SOTA Classifier Arch**: TimeDownsample1D + SEBlock + Transformer Encoder + **SoftmaxFocalLoss**

## V4 架构升级 (v27)

### 核心改进 (SOTA 对标分析)
1. **RoPE Attention**: 替换 Sinusoidal PE，如同 SongFormer 使用 Rotary Position Embedding
2. **单分支架构**: 移除 V3 的 3 分支 + 5 头加权融合，改为单分支 + 单头 (SOTA 风格)
3. **更强 Conv Frontend**: 3 层 Conv2D + BatchNorm + GELU (代替 V3 的 2 层 ReLU)
4. **更高容量**: d_model=256, feedforward=1024, 6 层 Transformer, 8 heads
5. **SOTA Input Projection**: Linear → LayerNorm → GELU → Dropout → Linear (同 SongFormer)
6. **SiLU Boundary Head**: MLP [128, 64, 8, 1] 使用 SiLU 激活 (同 SOTA Head)
7. **标签下采样**: 训练时自动将标签插值到模型输出尺寸
8. **推理上采样**: 滑动窗口推理时自动将下采样输出插值回原始分辨率

### V3 vs V4 对比
| 方面 | V3 (songformer_ds) | V4 |
|------|-------------------|----|
| d_model | 192 | **256** |
| Transformer 层数 | 4+2+2 (3分支) | **6** (单分支) |
| feedforward | 384 | **1024** |
| heads | 4 | **8** |
| Attention | MultiheadAttention | **RoPE MHA** |
| 预测头 | 5 头加权融合 | **单头 MLP** |
| 参数量 | ~3M | ~8M |
| 设计理念 | 多分支多尺度 | **SOTA 简洁设计** |

## 推荐训练命令（v27 - V4架构）

### V4 边界检测 (推荐)
```bash
python3 -m model.train_pipeline \
  --data_dir /root/MusicSeg/data/songform-hx-aligned \
  --ckpt_dir /root/MusicSeg/checkpoints/hx_boundary_v27 \
  --arch v4 \
  --epochs 300 \
  --boundary_epochs 300 \
  --classifier_epochs 0 \
  --boundary_batch_size 4 \
  --boundary_lr 1e-4 \
  --boundary_eval_mode paper \
  --boundary_eval_interval 1 \
  --boundary_eval_thresholds "0.0,0.003,0.005,0.008,0.01,0.015,0.02,0.03" \
  --boundary_loss bce \
  --boundary_contrastive_weight 0.05 \
  --boundary_peak_refine_radius 24 \
  --boundary_songformer_postprocess \
  --boundary_local_maxima_filter_size 3 \
  --boundary_postprocess_window_past_sec 6 \
  --boundary_postprocess_window_future_sec 6 \
  --boundary_postprocess_downsample_factor 3 \
  --boundary_pos_weight_max 100 \
  --boundary_tv_weight 0.05 \
  --boundary_rate_weight 0.0 \
  --boundary_weight_decay 3e-7 \
  --boundary_beta1 0.8 \
  --boundary_beta2 0.999 \
  --boundary_grad_clip 1.0 \
  --boundary_early_stopping_patience 50 \
  --boundary_accum_steps 4 \
  --boundary_warmup_steps 500 \
  --boundary_ema_decay 0.999 \
  --boundary_ema_update_after_steps 200
```

### V4 核心调优点
1. **纯 BCE Loss**: SOTA SongFormer 也使用 BCE 作为主损失，简单稳定
2. **更低学习率 1e-4**: 更大模型需要更小的学习率
3. **更低 weight_decay 3e-7**: 对标 SOTA 配置
4. **beta1=0.8**: 对标 SOTA 配置
5. **TV weight 0.05**: 对标 SOTA 的 boundary_tvloss_weight
6. **降低 contrastive weight**: 0.05 代替 0.1，避免干扰主损失
7. **batch_size=4**: 更大模型需要更小 batch + 更多梯度累积

## 旧版训练命令（v26_final_optimized - songformer_ds）
**核心优化点 (基于历次复盘)**:
1. **数据采样 (Data Sampling)**: 修正了过拟合的根源——95% 边界居中采样。调整为 **60% 边界 + 40% 随机背景**，大幅增加负样本多样性，降低误报率。
2. **数据增强 (SpecAugment)**: 增强了时间掩码 (Time Mask) 宽度 (35->100)，迫使模型利用更长上下文。
3. **架构升级 (Stochastic Depth)**: 引入 **DropPath (0.1)** 到 Transformer Block，这是深层网络防止过拟合的标配 (SOTA)。
4. **损失函数 (BCE+Dice)**: 引入 **Soft Dice Loss** 与 BCE 结合 (`bce_dice`)，直接优化 F1/IOU 指标，解决类别不平衡问题。
5. **正则化 (TV Loss)**: 保持微量 TV Loss (0.01) 以平滑输出。

**边界检测**
```bash
python3 -m model.train_pipeline \
  --data_dir /root/MusicSeg/data/songform-hx-aligned \
  --ckpt_dir /root/MusicSeg/checkpoints/hx_boundary_v26 \
  --arch songformer_ds \
  --epochs 300 \
  --boundary_epochs 300 \
  --classifier_epochs 0 \
  --boundary_batch_size 6 \
  --boundary_lr 2e-4 \
  --boundary_eval_mode paper \
  --boundary_eval_interval 1 \
  --boundary_eval_thresholds "0.0,0.003,0.005,0.008,0.01,0.015,0.02,0.03" \
  --boundary_loss bce_dice \
  --boundary_contrastive_weight 0.1 \
  --boundary_focal_alpha 0.75 \
  --boundary_peak_refine_radius 24 \
  --boundary_songformer_postprocess \
  --boundary_local_maxima_filter_size 3 \
  --boundary_postprocess_window_past_sec 6 \
  --boundary_postprocess_window_future_sec 6 \
  --boundary_postprocess_downsample_factor 3 \
  --boundary_pos_weight_max 100 \
  --boundary_tv_weight 0.01 \
  --boundary_rate_weight 0.05 \
  --boundary_weight_decay 1e-5 \
  --boundary_beta1 0.9 \
  --boundary_beta2 0.999 \
  --boundary_grad_clip 1.0 \
  --boundary_early_stopping_patience 50 \
  --boundary_accum_steps 4 \
  --boundary_warmup_steps 1000 \
  --boundary_ema_decay 0.999 \
  --boundary_ema_update_after_steps 200
```

**分类器**
```bash
python3 -m model.train_pipeline \
  --data_dir /root/MusicSeg/data/songform-hx-aligned \
  --ckpt_dir /root/MusicSeg/checkpoints/hx_boundary_v26 \
  --arch songformer_ds \
  --epochs 300 \
  --boundary_epochs 0 \
  --classifier_epochs 100 \
  --classifier_batch_size 12 \
  --classifier_lr 1.5e-4 \
  --classifier_weight_decay 1e-4 \
  --classifier_beta1 0.8 \
  --classifier_beta2 0.999 \
  --classifier_segment_frames 384 \
  --classifier_lr_schedule cosine \
  --classifier_warmup_steps 500 \
  --classifier_balance_samples \
  --classifier_patience 30 \
  --classifier_label_smoothing 0.1 \
  --classifier_hidden 768 \
  --classifier_loss focal \
  --classifier_weight_power 0.6 \
  --classifier_grad_clip 1.0 \
  --classifier_ema_decay 0.999
```

**注意**: 分类器 Loss 现已支持 `focal` (使用 v26 新增的 SoftmaxFocalLoss)。

## debug 指令（阈值扫参）
```bash
python3 -m model.eval_paper_metrics \
  --data_dir /root/MusicSeg/data/songform-hx-aligned \
  --boundary_ckpt /root/MusicSeg/checkpoints/hx_boundary_v26/boundary.pt \
  --classifier_ckpt /root/MusicSeg/checkpoints/hx_boundary_v26/classifier.pt \
  --split val \
  --no_beat_snap \
  --no_hierarchical \
  --sweep_thresholds "0.0,0.003,0.005,0.008,0.01,0.015,0.02,0.03" \
  --debug_songs 3
```

## 推荐测试指令（v26）

**重要提示**: 请确保本地代码已同步到服务器，否则会报 KeyError 或 ValueError。

**1. 边界检测模型测试**
```bash
python3 -m model.eval_paper_metrics \
  --data_dir /root/MusicSeg/data/songform-hx-aligned \
  --boundary_ckpt /root/MusicSeg/checkpoints/hx_boundary_v25_final/boundary.pt \
  --classifier_ckpt /root/MusicSeg/checkpoints/hx_boundary_v25_final/classifier.pt \
  --split test \
  --no_beat_snap \
  --no_hierarchical
```

**2. 分类器测试**
```bash
python3 test_classifier.py \
  --data_dir /root/MusicSeg/data/songform-hx-aligned \
  --ckpt_path /root/MusicSeg/checkpoints/hx_boundary_v25_final/classifier.pt
```

## 最近修改
- model/eval_paper_metrics.py (修复 split=test 为空时的崩溃问题，增加自动回退逻辑)
- test_classifier.py (修复 model_state_dict 加载兼容性，增加 split 回退逻辑)
- model/infer.py (增强 checkpoint 加载兼容性)
- model/train_classifier.py (修复 EMA 在 LongTensor 上的运行时错误)
- model/config.py（min_distance=48, label_radius=36, label_sigma=14.0）