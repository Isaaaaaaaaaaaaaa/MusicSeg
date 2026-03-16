import React from 'react';
import { History, FileText, Clock, CheckCircle, AlertTriangle, Calendar } from 'lucide-react';

const HistoryItem = ({ id, filename, date, duration, status, confidence }) => (
    <div className="flex items-center justify-between p-4 bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl hover:border-blue-300 dark:hover:border-cyber-primary/30 transition-all group hover:shadow-md">
        <div className="flex items-center gap-4">
            <div className={`p-3 rounded-lg ${
                status === 'completed' ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-yellow-100 dark:bg-yellow-500/10 text-yellow-600 dark:text-yellow-400'
            }`}>
                {status === 'completed' ? <CheckCircle className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
            </div>
            <div>
                <h4 className="font-bold text-gray-900 dark:text-white text-sm">{filename}</h4>
                <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-gray-500 dark:text-gray-400 font-mono flex items-center gap-1">
                        <Calendar className="w-3 h-3" /> {date}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 font-mono flex items-center gap-1">
                        <Clock className="w-3 h-3" /> {duration}
                    </span>
                </div>
            </div>
        </div>
        
        <div className="flex items-center gap-6">
            <div className="text-right hidden md:block">
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider">置信度</p>
                <p className="text-sm font-mono font-bold text-gray-900 dark:text-white">{confidence}</p>
            </div>
            <button className="p-2 hover:bg-gray-100 dark:hover:bg-white/10 rounded-lg text-gray-400 hover:text-blue-600 dark:hover:text-cyber-primary transition-colors">
                <FileText className="w-5 h-5" />
            </button>
        </div>
    </div>
);

const HistoryPage = () => {
    // Mock Data
    const historyData = [
        { id: 1, filename: "Cyberpunk City.mp3", date: "2024-03-15 14:30", duration: "3:45", status: "completed", confidence: "98.5%" },
        { id: 2, filename: "Synthwave Mix.mp3", date: "2024-03-15 13:15", duration: "4:20", status: "completed", confidence: "94.2%" },
        { id: 3, filename: "Night Drive.wav", date: "2024-03-14 09:45", duration: "2:55", status: "completed", confidence: "96.8%" },
        { id: 4, filename: "Retro Wave.flac", date: "2024-03-13 18:20", duration: "5:10", status: "completed", confidence: "92.1%" },
        { id: 5, filename: "Failed Audio.mp3", date: "2024-03-12 11:00", duration: "0:00", status: "failed", confidence: "0%" },
    ];

    return (
        <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8 animate-slide-up">
            <div className="flex items-center gap-4 mb-8">
                <div className="p-3 rounded-xl bg-purple-600 dark:bg-cyber-secondary text-white shadow-lg shadow-purple-500/20">
                    <History className="w-6 h-6" />
                </div>
                <div>
                    <h1 className="font-display text-3xl font-bold text-gray-900 dark:text-white">分析历史</h1>
                    <p className="text-gray-500 dark:text-gray-400 font-mono text-sm mt-1">查看过往的分析记录与报告</p>
                </div>
            </div>

            <div className="bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-2xl p-6 shadow-xl backdrop-blur-md">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="font-display text-lg font-bold text-gray-900 dark:text-white">最近记录</h3>
                    <div className="flex gap-2">
                        <select className="bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-xs font-mono text-gray-600 dark:text-gray-300 focus:outline-none focus:border-blue-500">
                            <option>所有状态</option>
                            <option>已完成</option>
                            <option>失败</option>
                        </select>
                        <select className="bg-gray-50 dark:bg-black/20 border border-gray-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-xs font-mono text-gray-600 dark:text-gray-300 focus:outline-none focus:border-blue-500">
                            <option>最近 7 天</option>
                            <option>最近 30 天</option>
                        </select>
                    </div>
                </div>

                <div className="space-y-3">
                    {historyData.map((item) => (
                        <HistoryItem key={item.id} {...item} />
                    ))}
                </div>
                
                <div className="mt-6 pt-6 border-t border-gray-200 dark:border-white/10 text-center">
                    <button className="text-sm font-mono font-bold text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-cyber-primary transition-colors uppercase tracking-widest">
                        加载更多记录
                    </button>
                </div>
            </div>
        </div>
    );
};

export default HistoryPage;
