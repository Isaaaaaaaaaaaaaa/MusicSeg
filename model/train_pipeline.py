import argparse
import os

from .train_with_metrics import train_boundary_with_metrics
from .train_songformer import train_songformer
from .train_classifier import train as train_classifier


def main():
    parser = argparse.ArgumentParser(
        description="使用标准音乐结构分析指标训练音乐结构分析模型"
    )
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--boundary_epochs", type=int, default=0, help="边界模型训练轮数（0则使用epochs）")
    parser.add_argument("--classifier_epochs", type=int, default=0, help="分类器训练轮数（0则使用epochs）")
    parser.add_argument(
        "--arch", choices=["lstm", "transformer", "ms_transformer", "ms_transformer_v2", "songformer_ds", "multi_res", "songformer"], default="transformer", help="模型架构"
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_label_count", type=int, default=20, help="最小标签数量")
    parser.add_argument("--min_seg_seconds", type=float, default=0.5, help="最小区段长度（秒）")
    parser.add_argument("--balance_samples", action="store_true", default=False, help="是否平衡样本")
    parser.add_argument("--max_train_items", type=int, default=0, help="最大训练样本数")
    parser.add_argument("--max_labels", type=int, default=12, help="最大标签数")
    parser.add_argument("--workers", type=int, default=None, help="数据加载工作线程数")
    parser.add_argument(
        "--no_functional_map",
        action="store_false",
        dest="use_functional_map",
        help="禁用功能性标签映射",
    )
    parser.add_argument(
        "--eval_tolerance",
        type=float,
        default=0.5,
        help="边界匹配容差（秒），默认0.5秒",
    )
    parser.add_argument("--boundary_batch_size", type=int, default=8, help="边界训练批次大小")
    parser.add_argument("--boundary_lr", type=float, default=1e-3, help="边界训练学习率")
    parser.add_argument("--boundary_eval_thresholds", default="", help="边界阈值列表，逗号分隔")
    parser.add_argument("--boundary_eval_interval", type=int, default=1, help="边界每隔N个epoch评估一次")
    parser.add_argument("--boundary_eval_mode", choices=["standard", "paper"], default="standard")
    parser.add_argument("--boundary_tv_weight", type=float, default=0.0)
    parser.add_argument("--boundary_sparsity_weight", type=float, default=0.0)
    parser.add_argument("--boundary_rate_weight", type=float, default=0.0)
    parser.add_argument("--boundary_weight_decay", type=float, default=0.01)
    parser.add_argument("--boundary_beta1", type=float, default=0.9)
    parser.add_argument("--boundary_beta2", type=float, default=0.999)
    parser.add_argument("--boundary_grad_clip", type=float, default=1.0)
    parser.add_argument("--boundary_warmup_steps", type=int, default=300)
    parser.add_argument("--boundary_ema_decay", type=float, default=0.999)
    parser.add_argument("--boundary_ema_update_after_steps", type=int, default=0)
    parser.add_argument("--boundary_ema_eval", action="store_true", default=False)
    parser.add_argument("--boundary_reg_ramp_epochs", type=int, default=4)
    parser.add_argument("--boundary_pos_weight_max", type=float, default=100.0)
    parser.add_argument("--boundary_loss", choices=["bce", "focal", "hybrid", "dice", "bce_dice"], default="bce")
    parser.add_argument("--boundary_focal_alpha", type=float, default=0.25)
    parser.add_argument("--boundary_focal_gamma", type=float, default=2.0)
    parser.add_argument("--boundary_focal_warmup_epochs", type=int, default=3)
    parser.add_argument("--boundary_focal_ramp_epochs", type=int, default=6)
    parser.add_argument("--boundary_contrastive_weight", type=float, default=0.1, help="结构对比损失权重")
    parser.add_argument("--boundary_peak_refine_radius", type=int, default=6)
    parser.add_argument("--boundary_songformer_postprocess", action="store_true", default=False)
    parser.add_argument("--boundary_local_maxima_filter_size", type=int, default=3)
    parser.add_argument("--boundary_postprocess_window_past_sec", type=float, default=12.0)
    parser.add_argument("--boundary_postprocess_window_future_sec", type=float, default=12.0)
    parser.add_argument("--boundary_postprocess_downsample_factor", type=int, default=3)
    parser.add_argument("--boundary_accum_steps", type=int, default=1)
    parser.add_argument("--boundary_early_stopping_patience", type=int, default=8, help="边界模型早停耐心")
    parser.add_argument("--songformer_warmup_steps", type=int, default=300, help="SongFormer warmup steps")
    parser.add_argument("--songformer_total_steps", type=int, default=0, help="SongFormer total steps")
    parser.add_argument("--songformer_max_seconds", type=float, default=420.0, help="SongFormer 最大时长（秒）")
    parser.add_argument("--songformer_downsample_factor", type=int, default=3, help="SongFormer 下采样倍数")
    parser.add_argument("--songformer_grad_accum", type=int, default=1, help="SongFormer 梯度累积步数")
    parser.add_argument("--songformer_no_amp", action="store_false", dest="songformer_amp")
    parser.set_defaults(songformer_amp=True)
    parser.add_argument("--audio_sample_rate", type=int, default=None, help="采样率覆盖")
    parser.add_argument("--audio_n_mels", type=int, default=None, help="梅尔频带数覆盖")
    parser.add_argument("--audio_hop_length", type=int, default=None, help="hop_length 覆盖")
    parser.add_argument("--audio_n_fft", type=int, default=None, help="n_fft 覆盖")
    parser.add_argument("--classifier_hidden", type=int, default=512, help="分类器隐藏层宽度")
    parser.add_argument("--classifier_batch_size", type=int, default=16, help="分类器训练批次大小")
    parser.add_argument("--classifier_lr", type=float, default=1e-3, help="分类器训练学习率")
    parser.add_argument("--classifier_weight_decay", type=float, default=0.0, help="分类器权重衰减")
    parser.add_argument("--classifier_beta1", type=float, default=0.9, help="分类器 Adam beta1")
    parser.add_argument("--classifier_beta2", type=float, default=0.999, help="分类器 Adam beta2")
    parser.add_argument("--classifier_weight_power", type=float, default=0.75, help="分类器类别权重指数")
    parser.add_argument("--classifier_patience", type=int, default=3, help="分类器早停耐心")
    parser.add_argument("--classifier_min_delta", type=float, default=0.0, help="分类器早停最小提升")
    parser.add_argument("--classifier_grad_clip", type=float, default=1.0, help="分类器梯度裁剪")
    parser.add_argument("--classifier_ema_decay", type=float, default=0.999, help="分类器EMA衰减")
    parser.add_argument("--parallel", action="store_true", default=False, help="并行训练边界和分类器")
    parser.add_argument("--classifier_input_type", choices=["pooled", "mel"], default="mel", help="分类器输入类型")
    parser.add_argument("--classifier_segment_frames", type=int, default=256, help="分类器段落帧长度")
    parser.add_argument("--classifier_lr_schedule", choices=["plateau", "cosine"], default="plateau", help="分类器学习率调度")
    parser.add_argument("--classifier_warmup_steps", type=int, default=0, help="分类器warmup步数")
    parser.add_argument("--classifier_total_steps", type=int, default=0, help="分类器总步数（0自动）")
    parser.add_argument("--classifier_balance_samples", action="store_true", default=False, help="分类器样本均衡采样")
    parser.add_argument("--classifier_loss", choices=["ce", "focal"], default="ce", help="分类器损失函数")
    parser.add_argument("--classifier_gamma", type=float, default=1.5, help="分类器 focal loss gamma")
    parser.add_argument("--classifier_label_smoothing", type=float, default=0.0, help="分类器标签平滑")
    parser.add_argument("--ckpt_dir", default="checkpoints", help="checkpoint 输出目录")
    parser.set_defaults(use_functional_map=True)
    args = parser.parse_args()

    print("=" * 60)
    print("开始训练音乐结构分析模型")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"边界检测架构: {args.arch}")
    print(f"评估容差: {args.eval_tolerance}s")
    print(f"使用功能性标签: {args.use_functional_map}")
    print(f"checkpoint 目录: {args.ckpt_dir}")
    print("=" * 60)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    boundary_out = os.path.join(args.ckpt_dir, "boundary.pt")
    classifier_out = os.path.join(args.ckpt_dir, "classifier.pt")
    songformer_out = os.path.join(args.ckpt_dir, "songformer.pt")

    boundary_thresholds = None
    if args.boundary_eval_thresholds:
        boundary_thresholds = []
        for item in str(args.boundary_eval_thresholds).split(","):
            item = item.strip()
            if item:
                boundary_thresholds.append(float(item))

    if args.parallel and args.arch != "songformer":
        print("\n【并行模式】启动并行训练...")
        import subprocess
        import sys
        
        # Construct commands
        cmd_base = [sys.executable, "-m", "model.train_pipeline"]
        
        # Pass through all arguments except --parallel
        args_list = []
        for key, value in vars(args).items():
            if key == "parallel" or value is None:
                continue
                
            # Special handling for store_false / dest args
            if key == "use_functional_map":
                if not value:
                    args_list.append("--no_functional_map")
                continue
            
            if key == "songformer_amp":
                if not value:
                    args_list.append("--songformer_no_amp")
                continue

            if isinstance(value, bool):
                if value:
                    args_list.append(f"--{key}")
            else:
                args_list.append(f"--{key}")
                args_list.append(str(value))
        
        # Boundary command (disable classifier training)
        cmd_boundary = cmd_base + args_list + ["--classifier_epochs", "0"]
        
        # Classifier command (disable boundary training)
        cmd_classifier = cmd_base + args_list + ["--boundary_epochs", "0"]
        
        print(f"启动边界训练进程 (PID将显示在下方)...")
        p1 = subprocess.Popen(cmd_boundary)
        
        print(f"启动分类器训练进程 (PID将显示在下方)...")
        p2 = subprocess.Popen(cmd_classifier)
        
        try:
            exit_code1 = p1.wait()
            exit_code2 = p2.wait()
            if exit_code1 == 0 and exit_code2 == 0:
                print("\n所有并行任务完成！")
            else:
                print(f"\n任务失败: Boundary={exit_code1}, Classifier={exit_code2}")
        except KeyboardInterrupt:
            p1.terminate()
            p2.terminate()
            print("\n并行训练已终止")
        return

    boundary_epochs = int(args.boundary_epochs)
    classifier_epochs = int(args.classifier_epochs)

    # Only default to epochs if both are 0, OR if specific epochs are not provided but intended to run
    # However, to fix the issue where boundary runs when boundary_epochs=0:
    # We should strictly respect the 0 value.
    # Logic:
    # If boundary_epochs is set (even 0), use it.
    # If not set (default 0) AND classifier_epochs is also 0, then maybe use global epochs?
    # Actually, the previous logic was:
    # boundary_epochs = int(args.boundary_epochs) if int(args.boundary_epochs) > 0 else int(args.epochs)
    # This forces boundary_epochs to be args.epochs if user passes 0. This is the BUG.
    
    # New Logic:
    # If user explicitly wants to run something, they should set it > 0.
    # If they rely on default (0), and global epochs > 0, we need to decide what to run.
    # Usually: run both if both are 0.
    
    # But here we want explicit control.
    # If user passed --boundary_epochs 0, they mean 0.
    # But argparse default is 0. How to distinguish "user passed 0" vs "default 0"?
    # We can't easily.
    
    # Better logic:
    # If boundary_epochs == 0 and classifier_epochs == 0:
    #    boundary_epochs = args.epochs
    #    classifier_epochs = args.epochs
    # Else:
    #    Use provided values (even if 0)
    
    if args.boundary_epochs == 0 and args.classifier_epochs == 0:
        boundary_epochs = args.epochs
        classifier_epochs = args.epochs
    else:
        boundary_epochs = args.boundary_epochs
        classifier_epochs = args.classifier_epochs
    
    if boundary_epochs > 0:
        if args.arch == "songformer":
            print("\n【阶段1】训练 SongFormer（边界 + 功能标签联合）")
            train_songformer(
                args.data_dir,
                songformer_out,
                boundary_epochs,
                args.boundary_batch_size,
                args.boundary_lr,
                args.songformer_warmup_steps,
                args.songformer_total_steps,
                args.seed,
                args.songformer_max_seconds,
                args.songformer_downsample_factor,
                args.boundary_eval_interval,
                args.songformer_grad_accum,
                args.songformer_amp,
                args.audio_sample_rate,
                args.audio_n_mels,
                args.audio_hop_length,
                args.audio_n_fft,
            )
        else:
            print("\n【阶段1】训练边界检测模型")
            train_boundary_with_metrics(
                data_dir=args.data_dir,
                out_path=boundary_out,
                epochs=boundary_epochs,
                batch_size=args.boundary_batch_size,
                lr=args.boundary_lr,
                arch=args.arch,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                eval_tolerance=args.eval_tolerance,
                eval_thresholds=boundary_thresholds,
                eval_interval=args.boundary_eval_interval,
                eval_mode=args.boundary_eval_mode,
                tv_weight=args.boundary_tv_weight,
                sparsity_weight=args.boundary_sparsity_weight,
                rate_weight=args.boundary_rate_weight,
                weight_decay=args.boundary_weight_decay,
                beta1=args.boundary_beta1,
                beta2=args.boundary_beta2,
                grad_clip=args.boundary_grad_clip,
                warmup_steps=args.boundary_warmup_steps,
                ema_decay=args.boundary_ema_decay,
                ema_update_after_steps=args.boundary_ema_update_after_steps,
                ema_eval=args.boundary_ema_eval,
                reg_ramp_epochs=args.boundary_reg_ramp_epochs,
                pos_weight_max=args.boundary_pos_weight_max,
                boundary_loss=args.boundary_loss,
                focal_alpha=args.boundary_focal_alpha,
                focal_gamma=args.boundary_focal_gamma,
                focal_warmup_epochs=args.boundary_focal_warmup_epochs,
                focal_ramp_epochs=args.boundary_focal_ramp_epochs,
                early_stopping_patience=args.boundary_early_stopping_patience,
                contrastive_weight=args.boundary_contrastive_weight,
                peak_refine_radius=args.boundary_peak_refine_radius,
                use_songformer_postprocess=args.boundary_songformer_postprocess,
                local_maxima_filter_size=args.boundary_local_maxima_filter_size,
                postprocess_window_past_sec=args.boundary_postprocess_window_past_sec,
                postprocess_window_future_sec=args.boundary_postprocess_window_future_sec,
                postprocess_downsample_factor=args.boundary_postprocess_downsample_factor,
                accum_steps=args.boundary_accum_steps,
                audio_sample_rate=args.audio_sample_rate,
                audio_n_mels=args.audio_n_mels,
                audio_hop_length=args.audio_hop_length,
                audio_n_fft=args.audio_n_fft,
            )

    if classifier_epochs > 0 and args.arch != "songformer":
        print("\n【阶段2】训练段落分类器模型")
        train_classifier(
            data_dir=args.data_dir,
            out_path=classifier_out,
            epochs=classifier_epochs,
            batch_size=args.classifier_batch_size,
            lr=args.classifier_lr,
            weight_decay=args.classifier_weight_decay,
            beta1=args.classifier_beta1,
            beta2=args.classifier_beta2,
            loss_name=args.classifier_loss,
            gamma=args.classifier_gamma,
            label_smoothing=args.classifier_label_smoothing,
            use_weights=True,
            weight_power=args.classifier_weight_power,
            hidden=args.classifier_hidden,
            patience=args.classifier_patience,
            min_delta=args.classifier_min_delta,
            grad_clip=args.classifier_grad_clip,
            input_type=args.classifier_input_type,
            segment_frames=args.classifier_segment_frames,
            lr_schedule=args.classifier_lr_schedule,
            warmup_steps=args.classifier_warmup_steps,
            total_steps=args.classifier_total_steps,
            balance_samples=args.classifier_balance_samples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_label_count=args.min_label_count,
            min_seg_seconds=args.min_seg_seconds,
            max_train_items=args.max_train_items,
            max_labels=args.max_labels,
            workers=args.workers,
            use_functional_map=args.use_functional_map,
            audio_sample_rate=args.audio_sample_rate,
            audio_n_mels=args.audio_n_mels,
            audio_hop_length=args.audio_hop_length,
            audio_n_fft=args.audio_n_fft,
            ema_decay=args.classifier_ema_decay,
        )

    print("\n" + "=" * 60)
    print("训练完成！")
    if args.arch == "songformer":
        print(f"模型保存在: {songformer_out}")
    else:
        print(f"模型保存在: {boundary_out} 和 {classifier_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
