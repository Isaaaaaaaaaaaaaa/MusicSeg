import React, { useRef } from 'react';
import { Activity, Music, Upload, FileText, Moon, Sun, Home, BookOpen, Grid, Settings as SettingsIcon, LayoutDashboard, History } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const Sidebar = ({ songs, currentSong, onSelectSong, onUpload, isUploading, onNavigate, currentView }) => {
    const fileInputRef = useRef(null);
    const { theme, toggleTheme } = useTheme();

    const NavItem = ({ icon: Icon, label, view, onClick }) => (
        <button 
            onClick={onClick || (() => onNavigate(view))}
            className={`w-full text-left px-4 py-3 rounded-lg transition-all flex items-center gap-3 group ${
                currentView === view
                ? 'bg-blue-50 dark:bg-white/5 text-blue-600 dark:text-white font-bold shadow-sm' 
                : 'text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-white/5 hover:text-gray-900 dark:hover:text-white'
            }`}
        >
            <Icon className={`w-4 h-4 ${currentView === view ? 'text-blue-600 dark:text-cyber-primary' : 'text-gray-400 dark:text-gray-500 group-hover:text-gray-600 dark:group-hover:text-gray-300'}`} />
            <span className="text-sm font-medium">{label}</span>
            {currentView === view && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-600 dark:bg-cyber-primary animate-pulse"></div>
            )}
        </button>
    );

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            onUpload(e.target.files[0]);
        }
    };

    return (
        <div className="w-64 h-full bg-white dark:bg-cyber-panel border-r border-gray-200 dark:border-white/5 flex flex-col backdrop-blur-md transition-colors duration-300 shadow-lg dark:shadow-none z-20">
            <div className="p-6 border-b border-gray-200 dark:border-white/5 flex justify-between items-center">
                <div className="flex items-center gap-2 cursor-pointer" onClick={() => onNavigate('landing')}>
                    <div className="w-8 h-8 rounded-lg bg-blue-600 dark:bg-cyber-primary flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <Activity className="w-5 h-5 text-white dark:text-black" />
                    </div>
                    <div>
                        <h2 className="font-display text-lg font-bold text-gray-900 dark:text-white tracking-tight leading-none">
                            MusicSeg
                        </h2>
                        <p className="text-gray-500 dark:text-cyber-dim text-[10px] font-mono tracking-widest uppercase">AI 智能分析</p>
                    </div>
                </div>
            </div>
            
            <div className="p-4 space-y-1">
                <div className="px-4 py-2 text-[10px] font-mono font-bold text-gray-400 dark:text-cyber-dim uppercase tracking-[0.2em] mb-2">主菜单</div>
                <NavItem icon={LayoutDashboard} label="仪表盘" view="dashboard" />
                <NavItem icon={Grid} label="媒体库" view="library" />
                <NavItem icon={Activity} label="结构分析" view="app" />
                <NavItem icon={History} label="分析历史" view="history" />
                <NavItem icon={SettingsIcon} label="系统设置" view="settings" />
            </div>

            <div className="p-4 border-t border-gray-200 dark:border-white/5 space-y-4">
                <div className="space-y-2">
                    <div className="px-2 text-[10px] font-mono font-bold text-gray-500 dark:text-cyber-dim uppercase tracking-[0.2em]">快速操作</div>
                    <button 
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-cyber-primary dark:to-cyber-secondary hover:from-blue-700 hover:to-indigo-700 border border-transparent text-white dark:text-black transition-all flex items-center justify-center gap-2 font-bold text-sm group rounded-lg shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 hover:-translate-y-0.5"
                    >
                        {isUploading ? (
                            <span className="animate-pulse flex items-center gap-2">
                                <Activity className="w-4 h-4 animate-spin" />
                                上传中...
                            </span>
                        ) : (
                            <>
                                <Upload className="w-4 h-4 group-hover:-translate-y-1 transition-transform" />
                                上传音频
                            </>
                        )}
                    </button>
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        onChange={handleFileChange} 
                        accept="audio/*" 
                        className="hidden" 
                    />
                </div>
            </div>

            <div className="mt-auto p-4 border-t border-gray-200 dark:border-white/5 space-y-1">
                <NavItem icon={BookOpen} label="技术文档" view="intro" />
                <NavItem icon={Home} label="返回首页" view="landing" />
            </div>
        </div>
    );
};

export default Sidebar;
