import React from 'react';
import { ChevronRight, Activity, FileText, PlayCircle } from 'lucide-react';

const LandingPage = ({ onEnterApp, onEnterIntro }) => {
    return (
        <div className="relative w-full h-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-cyber-black overflow-hidden transition-colors duration-300">
            {/* Background Effects */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-100 via-gray-50 to-white dark:from-cyber-panel dark:via-cyber-black dark:to-black opacity-50"></div>
            <div className="absolute inset-0 noise-overlay"></div>
            
            {/* Hero Content */}
            <div className="z-10 text-center space-y-12 animate-slide-up px-4 max-w-6xl w-full">
                <div className="space-y-4">
                    <h1 className="font-display text-5xl md:text-8xl font-bold tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-500">
                        MusicSeg <span className="text-blue-600 dark:text-cyber-primary">AI</span>
                    </h1>
                    <p className="font-sans text-gray-600 dark:text-cyber-text text-lg md:text-xl tracking-wide max-w-2xl mx-auto opacity-80">
                        新一代音乐结构智能分析系统
                    </p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full">
                    {/* Experience Card */}
                    <button 
                        onClick={onEnterApp}
                        className="group relative p-8 bg-white/50 dark:bg-white/5 border border-gray-200 dark:border-white/10 hover:border-blue-500 dark:hover:border-cyber-primary rounded-xl transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 text-left flex flex-col items-start overflow-hidden backdrop-blur-sm"
                    >
                        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-cyber-primary/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                            <PlayCircle className="w-6 h-6 text-blue-600 dark:text-cyber-primary" />
                        </div>
                        <h3 className="font-display text-2xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-cyber-primary transition-colors">
                            体验模型
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed mb-6">
                            上传音频文件，实时可视化波形，体验 SOTA 级的音乐结构分割与段落分类能力。
                        </p>
                        <span className="mt-auto flex items-center gap-2 text-sm font-mono font-bold text-blue-600 dark:text-cyber-primary tracking-widest">
                            ENTER SYSTEM <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </span>
                    </button>

                    {/* Intro Card */}
                    <button 
                        onClick={onEnterIntro}
                        className="group relative p-8 bg-white/50 dark:bg-white/5 border border-gray-200 dark:border-white/10 hover:border-purple-500 dark:hover:border-cyber-secondary rounded-xl transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 text-left flex flex-col items-start overflow-hidden backdrop-blur-sm"
                    >
                        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        <div className="w-12 h-12 rounded-full bg-purple-100 dark:bg-cyber-secondary/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                            <FileText className="w-6 h-6 text-purple-600 dark:text-cyber-secondary" />
                        </div>
                        <h3 className="font-display text-2xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-purple-600 dark:group-hover:text-cyber-secondary transition-colors">
                            架构介绍
                        </h3>
                        <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed mb-6">
                            深入了解 SongFormer v26 架构细节，包括多尺度变换器、随机深度与 Focal Loss 训练策略。
                        </p>
                        <span className="mt-auto flex items-center gap-2 text-sm font-mono font-bold text-purple-600 dark:text-cyber-secondary tracking-widest">
                            READ DOCS <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </span>
                    </button>
                </div>
            </div>
            
            {/* Footer */}
            <div className="absolute bottom-8 text-center text-gray-400 dark:text-cyber-dim text-xs font-mono tracking-widest opacity-50">
                POWERED BY SONGFORMER V26 ARCHITECTURE
            </div>
        </div>
    );
};

export default LandingPage;
