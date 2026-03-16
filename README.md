# MusicSeg - SOTA 音乐结构智能分析系统

MusicSeg 是一个基于深度学习的音乐结构分析系统，集成了最新的 SongFormer v26 架构，能够实现高精度的音乐边界检测与段落分类（如 Intro, Verse, Chorus 等）。本项目包含完整的模型训练流水线、FastAPI 后端服务以及基于 React 的现代化前端界面。

## 目录结构

```
MS-main/
├── backend/            # 后端服务 (FastAPI)
│   ├── server.py       # 服务入口
│   └── ...
├── frontend/           # 前端界面 (React + Vite)
│   ├── src/            # 源代码
│   └── dist/           # 构建产物
├── model/              # 模型定义与训练脚本
│   ├── train_pipeline.py # 训练主入口
│   ├── model.py        # 模型架构定义
│   └── ...
└── scripts/            # 辅助脚本
```

## 环境准备

### 1. Python 环境 (后端 & 模型)

建议使用 Python 3.8+。

```bash
# 安装依赖
pip install -r backend/requirements.txt
```

*注意：如果需要训练模型，请确保安装了 PyTorch (GPU版本推荐)。*

### 2. Node.js 环境 (前端)

建议使用 Node.js 16+。

```bash
cd frontend
npm install
```

---

## 模型训练

使用 `model.train_pipeline` 模块进行模型训练。

### 基础训练指令

```bash
# 在 MS-main 根目录下运行
python -m model.train_pipeline \
  --data_dir /path/to/your/dataset \
  --epochs 50 \
  --arch songformer \
  --batch_size 8
```

### 关键参数说明

- `--data_dir`: 数据集路径（必须包含 aligned 格式的数据）
- `--arch`: 模型架构，推荐使用 `songformer` 或 `songformer_ds` (Deep Supervision)
- `--epochs`: 训练轮数
- `--boundary_eval_mode`: 评估模式 (`standard` 或 `paper`)

---

## 系统部署与运行

本系统采用前后端分离开发，静态资源整合部署的模式。

### 1. 前端构建

首先需要编译前端资源，生成的静态文件将被放置在 `frontend/dist/` 目录下，后端会自动挂载该目录。

```bash
cd frontend
npm run build
```

### 2. 启动后端服务

后端服务基于 FastAPI，负责处理模型推理请求并提供前端静态页面。

```bash
# 回到 MS-main 根目录
cd .. 

# 启动服务
python backend/server.py
```

*注意：`server.py` 内部配置了相对导入，请确保在 `backend/` 的父目录（即 `MS-main`）下通过 `python backend/server.py` 或模块方式运行，或者确保 `PYTHONPATH` 包含当前目录。*

### 3. 访问系统

服务启动后，打开浏览器访问：

```
http://localhost:8000
```

---

## 功能特性

- **可视化分析**: 实时波形显示，彩色标记音乐段落。
- **SOTA 模型**: 集成 SongFormer v26，支持 Deep Supervision 和 Stochastic Depth。
- **多维度展示**: 提供仪表盘 (Dashboard)、媒体库 (Library) 和详细的分析历史。
- **全中文界面**: 深度本地化，支持明亮/深色主题切换。

## 开发指南

- **前端开发**: `cd frontend && npm run dev` (默认端口 5173，需配置代理指向后端 8000)
- **后端开发**: 修改 `backend/server.py` 后需重启服务生效。
# MusicSeg
