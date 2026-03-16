import React from 'react';
import { ChevronRight, CheckCircle, Activity, AlertTriangle } from 'lucide-react';

const ModelIntroPage = ({ onBack }) => {
    return (
        <div className="flex-1 flex flex-col h-full bg-gray-50 dark:bg-cyber-black text-gray-900 dark:text-white overflow-y-auto relative p-12 transition-colors duration-300">
            <div className="absolute inset-0 noise-overlay fixed"></div>
            
            <button 
                onClick={onBack}
                className="absolute top-8 left-8 flex items-center gap-2 text-gray-500 dark:text-cyber-dim hover:text-blue-600 dark:hover:text-cyber-primary transition-colors font-mono text-xs tracking-widest z-20"
            >
                <ChevronRight className="w-4 h-4 rotate-180" /> 返回系统
            </button>

            <div className="max-w-4xl mx-auto z-10 w-full space-y-16 pb-20">
                {/* Header */}
                <div className="space-y-4 border-b border-gray-200 dark:border-white/10 pb-8">
                    <h1 className="font-display text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-gray-900 to-gray-500 dark:from-white dark:to-gray-400">
                        MusicSeg v26 架构详解
                    </h1>
                    <p className="font-mono text-blue-600 dark:text-cyber-primary text-sm tracking-widest">SOTA 级边界检测与分类系统</p>
                </div>

                {/* Section 1: Overview */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    <div className="space-y-6">
                        <h2 className="font-display text-2xl font-bold text-gray-900 dark:text-white">系统概览</h2>
                        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                            MusicSeg v26 采用了一种创新的<span className="text-blue-600 dark:text-cyber-secondary font-bold">多尺度变换器 (Multi-Scale Transformer)</span> 架构，
                            专门针对音乐结构的层级特性进行了优化。通过结合卷积神经网络的局部特征提取能力与Transformer的全局上下文建模能力，
                            该模型在边界检测精度（F1-Score）上达到了行业领先水平。
                        </p>
                        <div className="flex items-center gap-4 text-sm font-mono text-gray-500 dark:text-gray-500">
                            <span className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-blue-600 dark:text-cyber-primary" /> 深层监督学习</span>
                            <span className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-blue-600 dark:text-cyber-primary" /> 随机深度优化</span>
                        </div>
                    </div>
                    
                    {/* Stats Card */}
                    <div className="bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 p-6 rounded-lg space-y-4 backdrop-blur-sm shadow-lg dark:shadow-none">
                        <div className="space-y-2">
                            <div className="flex justify-between items-end">
                                <span className="text-sm text-gray-500 dark:text-gray-400">边界准确率 (0.5s)</span>
                                <span className="font-mono text-xl text-gray-900 dark:text-white">82.4%</span>
                            </div>
                            <div className="w-full h-1 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-600 dark:bg-cyber-primary w-[82.4%]"></div>
                            </div>
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between items-end">
                                <span className="text-sm text-gray-500 dark:text-gray-400">边界 F1 (3s)</span>
                                <span className="font-mono text-xl text-gray-900 dark:text-white">78.2%</span>
                            </div>
                            <div className="w-full h-1 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full bg-purple-600 dark:bg-cyber-secondary w-[78.2%]"></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Section 2: Architecture */}
                <div className="space-y-8">
                    <h2 className="font-display text-2xl font-bold text-gray-900 dark:text-white">技术架构</h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {/* Card 1 */}
                        <div className="bg-white dark:bg-transparent dark:glass p-6 rounded-lg border border-gray-200 dark:border-white/5 hover:border-blue-400 dark:hover:border-cyber-primary/30 transition-all group shadow-md dark:shadow-none">
                            <div className="w-10 h-10 rounded-full bg-blue-50 dark:bg-cyber-primary/10 flex items-center justify-center mb-4 group-hover:bg-blue-100 dark:group-hover:bg-cyber-primary/20">
                                <Activity className="w-5 h-5 text-blue-600 dark:text-cyber-primary" />
                            </div>
                            <h3 className="font-display text-lg font-bold mb-2">SongFormer 主干</h3>
                            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                                使用 TimeDownsample1D 模块进行高效的时域降采样，配合深层 Transformer 编码器捕获长距离依赖。
                            </p>
                        </div>

                        {/* Card 2 */}
                        <div className="bg-white dark:bg-transparent dark:glass p-6 rounded-lg border border-gray-200 dark:border-white/5 hover:border-purple-400 dark:hover:border-cyber-secondary/30 transition-all group shadow-md dark:shadow-none">
                            <div className="w-10 h-10 rounded-full bg-purple-50 dark:bg-cyber-secondary/10 flex items-center justify-center mb-4 group-hover:bg-purple-100 dark:group-hover:bg-cyber-secondary/20">
                                <AlertTriangle className="w-5 h-5 text-purple-600 dark:text-cyber-secondary" />
                            </div>
                            <h3 className="font-display text-lg font-bold mb-2">随机深度 (Stochastic Depth)</h3>
                            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                                引入 DropPath 技术，在训练过程中随机丢弃部分残差路径，有效防止过拟合，提升模型泛化能力。
                            </p>
                        </div>

                        {/* Card 3 */}
                        <div className="bg-white dark:bg-transparent dark:glass p-6 rounded-lg border border-gray-200 dark:border-white/5 hover:border-indigo-400 dark:hover:border-purple-500/30 transition-all group shadow-md dark:shadow-none">
                            <div className="w-10 h-10 rounded-full bg-indigo-50 dark:bg-purple-500/10 flex items-center justify-center mb-4 group-hover:bg-indigo-100 dark:group-hover:bg-purple-500/20">
                                <CheckCircle className="w-5 h-5 text-indigo-600 dark:text-purple-500" />
                            </div>
                            <h3 className="font-display text-lg font-bold mb-2">Softmax Focal Loss</h3>
                            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                                针对类别不平衡问题，采用 Focal Loss 变体，动态调整难易样本权重，使模型更关注难以分类的边界段。
                            </p>
                        </div>
                    </div>
                </div>

                {/* Section 3: Training */}
                <div className="border-t border-gray-200 dark:border-white/10 pt-8">
                    <h2 className="font-display text-2xl font-bold text-gray-900 dark:text-white mb-6">训练策略</h2>
                    <ul className="space-y-4 font-mono text-sm text-gray-600 dark:text-gray-400">
                        <li className="flex items-start gap-3">
                            <span className="text-blue-600 dark:text-cyber-primary mt-1">01.</span>
                            <span>使用 <span className="text-gray-900 dark:text-white font-bold">Deep Supervision</span> 技术，在多个尺度上同时计算辅助损失，加速收敛。</span>
                        </li>
                        <li className="flex items-start gap-3">
                            <span className="text-blue-600 dark:text-cyber-primary mt-1">02.</span>
                            <span>引入 <span className="text-gray-900 dark:text-white font-bold">TVLoss1D</span> (Total Variation Loss) 平滑输出概率，减少噪声波动。</span>
                        </li>
                        <li className="flex items-start gap-3">
                            <span className="text-blue-600 dark:text-cyber-primary mt-1">03.</span>
                            <span>采用 <span className="text-gray-900 dark:text-white font-bold">Exponential Moving Average (EMA)</span> 策略，稳定模型权重，提升最终性能。</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default ModelIntroPage;
