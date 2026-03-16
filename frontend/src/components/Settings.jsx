import React from 'react';
import { Settings as SettingsIcon, Moon, Sun, Bell, Shield, Cpu, Volume2, Database, Globe, Sliders } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const SettingSection = ({ title, children }) => (
    <div className="space-y-4">
        <h3 className="font-display text-lg font-bold text-gray-900 dark:text-white border-b border-gray-200 dark:border-white/10 pb-2 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-blue-600 dark:text-cyber-primary" />
            {title}
        </h3>
        <div className="space-y-4">
            {children}
        </div>
    </div>
);

const SettingItem = ({ icon: Icon, title, description, action }) => (
    <div className="flex items-center justify-between p-4 rounded-xl bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 hover:border-blue-300 dark:hover:border-cyber-primary/30 transition-all shadow-sm hover:shadow-md">
        <div className="flex items-center gap-4">
            <div className="p-2 rounded-lg bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-300">
                <Icon className="w-5 h-5" />
            </div>
            <div>
                <h4 className="font-bold text-gray-900 dark:text-white text-sm">{title}</h4>
                <p className="text-xs text-gray-500 dark:text-gray-400">{description}</p>
            </div>
        </div>
        <div>
            {action}
        </div>
    </div>
);

const Toggle = ({ checked, onChange }) => (
    <button 
        onClick={() => onChange(!checked)}
        className={`w-12 h-6 rounded-full p-1 transition-colors ${checked ? 'bg-blue-600 dark:bg-cyber-primary' : 'bg-gray-300 dark:bg-white/20'}`}
    >
        <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${checked ? 'translate-x-6' : 'translate-x-0'}`}></div>
    </button>
);

const Settings = () => {
    const { theme, toggleTheme } = useTheme();

    return (
        <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8 animate-slide-up">
            <div className="flex items-center gap-4 mb-8">
                <div className="p-3 rounded-xl bg-gray-900 dark:bg-white text-white dark:text-black shadow-lg">
                    <SettingsIcon className="w-6 h-6" />
                </div>
                <div>
                    <h1 className="font-display text-3xl font-bold text-gray-900 dark:text-white">系统设置</h1>
                    <p className="text-gray-500 dark:text-gray-400 font-mono text-sm mt-1">配置系统偏好与参数</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <SettingSection title="外观设置">
                    <SettingItem 
                        icon={theme === 'dark' ? Moon : Sun}
                        title="深色模式"
                        description="在明亮和深色主题之间切换"
                        action={<Toggle checked={theme === 'dark'} onChange={toggleTheme} />}
                    />
                    <SettingItem 
                        icon={Volume2}
                        title="交互音效"
                        description="启用界面点击和操作音效"
                        action={<Toggle checked={true} onChange={() => {}} />}
                    />
                    <SettingItem 
                        icon={Globe}
                        title="语言"
                        description="当前语言: 简体中文"
                        action={<span className="text-xs font-mono font-bold text-gray-500 dark:text-gray-400">ZH-CN</span>}
                    />
                </SettingSection>

                <SettingSection title="分析引擎">
                    <SettingItem 
                        icon={Cpu}
                        title="深度监督 (Deep Supervision)"
                        description="启用多尺度损失计算以提高边界精度"
                        action={<Toggle checked={true} onChange={() => {}} />}
                    />
                    <SettingItem 
                        icon={Shield}
                        title="安全模式"
                        description="在处理过程中加密音频数据"
                        action={<Toggle checked={false} onChange={() => {}} />}
                    />
                </SettingSection>

                <SettingSection title="数据与存储">
                    <SettingItem 
                        icon={Database}
                        title="自动保存结果"
                        description="分析完成后自动保存到本地存储"
                        action={<Toggle checked={true} onChange={() => {}} />}
                    />
                    <SettingItem 
                        icon={Bell}
                        title="通知提醒"
                        description="当批量分析任务完成时通知我"
                        action={<Toggle checked={true} onChange={() => {}} />}
                    />
                </SettingSection>
            </div>
        </div>
    );
};

export default Settings;
