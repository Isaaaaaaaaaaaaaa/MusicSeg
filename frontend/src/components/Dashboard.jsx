import React from 'react';
import { Activity, Clock, Music, Zap } from 'lucide-react';

const StatCard = ({ icon: Icon, label, value, trend, color }) => (
    <div className={`p-6 rounded-2xl border bg-white/50 dark:bg-white/5 backdrop-blur-md transition-all hover:scale-105 ${
        color === 'blue' ? 'border-blue-200 dark:border-blue-500/20 hover:border-blue-400 dark:hover:border-blue-500/40' :
        color === 'purple' ? 'border-purple-200 dark:border-purple-500/20 hover:border-purple-400 dark:hover:border-purple-500/40' :
        color === 'green' ? 'border-green-200 dark:border-green-500/20 hover:border-green-400 dark:hover:border-green-500/40' :
        'border-orange-200 dark:border-orange-500/20 hover:border-orange-400 dark:hover:border-orange-500/40'
    }`}>
        <div className="flex items-start justify-between mb-4">
            <div className={`p-3 rounded-xl ${
                color === 'blue' ? 'bg-blue-100 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400' :
                color === 'purple' ? 'bg-purple-100 dark:bg-purple-500/10 text-purple-600 dark:text-purple-400' :
                color === 'green' ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' :
                'bg-orange-100 dark:bg-orange-500/10 text-orange-600 dark:text-orange-400'
            }`}>
                <Icon className="w-6 h-6" />
            </div>
            {trend && (
                <span className={`text-xs font-mono px-2 py-1 rounded-full ${
                    trend > 0 ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-red-100 dark:bg-red-500/10 text-red-600 dark:text-red-400'
                }`}>
                    {trend > 0 ? '+' : ''}{trend}%
                </span>
            )}
        </div>
        <div className="space-y-1">
            <h3 className="text-2xl font-display font-bold text-gray-900 dark:text-white">{value}</h3>
            <p className="text-sm font-mono text-gray-500 dark:text-gray-400">{label}</p>
        </div>
    </div>
);

const RecentActivity = ({ activities }) => (
    <div className="col-span-1 lg:col-span-2 p-6 rounded-2xl border border-gray-200 dark:border-white/10 bg-white/50 dark:bg-white/5 backdrop-blur-md">
        <h3 className="font-display text-lg font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600 dark:text-cyber-primary" />
            Recent Activity
        </h3>
        <div className="space-y-4">
            {activities.map((activity, idx) => (
                <div key={idx} className="flex items-center gap-4 p-3 rounded-xl hover:bg-gray-100 dark:hover:bg-white/5 transition-colors group">
                    <div className="w-10 h-10 rounded-full bg-gray-100 dark:bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform">
                        {activity.type === 'upload' ? <Music className="w-4 h-4 text-blue-500" /> : <Zap className="w-4 h-4 text-purple-500" />}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">{activity.title}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">{activity.time}</p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-mono ${
                        activity.status === 'completed' ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-yellow-100 dark:bg-yellow-500/10 text-yellow-600 dark:text-yellow-400'
                    }`}>
                        {activity.status}
                    </div>
                </div>
            ))}
        </div>
    </div>
);

const Dashboard = ({ songs }) => {
    // Mock data based on props
    const totalSongs = songs.length;
    const analyzedSongs = songs.filter(s => s.analyzed).length || Math.floor(totalSongs * 0.4); // Mock
    const totalDuration = "4h 23m"; // Mock
    const avgConfidence = "94.2%"; // Mock

    const activities = [
        { type: 'analysis', title: 'Analyzed "Cyberpunk City.mp3"', time: '2 mins ago', status: 'completed' },
        { type: 'upload', title: 'Uploaded "Night Drive.wav"', time: '15 mins ago', status: 'completed' },
        { type: 'analysis', title: 'Analyzed "Synthwave Mix.mp3"', time: '1 hour ago', status: 'completed' },
        { type: 'upload', title: 'Uploaded "Retro Wave.flac"', time: '3 hours ago', status: 'completed' },
    ];

    return (
        <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8 animate-slide-up">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
                    <p className="text-gray-500 dark:text-gray-400 font-mono text-sm mt-1">System Overview & Statistics</p>
                </div>
                <div className="text-right">
                    <p className="font-mono text-xs text-gray-400 dark:text-gray-500">LAST SYNC</p>
                    <p className="font-mono text-sm text-gray-900 dark:text-white">{new Date().toLocaleTimeString()}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard icon={Music} label="Total Songs" value={totalSongs} trend={12} color="blue" />
                <StatCard icon={Activity} label="Analyzed" value={analyzedSongs} trend={5} color="purple" />
                <StatCard icon={Clock} label="Total Duration" value={totalDuration} color="green" />
                <StatCard icon={Zap} label="Avg. Confidence" value={avgConfidence} trend={2.4} color="orange" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <RecentActivity activities={activities} />
                
                {/* System Status Card */}
                <div className="p-6 rounded-2xl border border-gray-200 dark:border-white/10 bg-white/50 dark:bg-white/5 backdrop-blur-md">
                    <h3 className="font-display text-lg font-bold text-gray-900 dark:text-white mb-6">System Status</h3>
                    <div className="space-y-6">
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-500 dark:text-gray-400">CPU Usage</span>
                                <span className="text-gray-900 dark:text-white font-mono">24%</span>
                            </div>
                            <div className="h-1.5 bg-gray-200 dark:bg-white/10 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500 w-[24%] rounded-full"></div>
                            </div>
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-500 dark:text-gray-400">Memory</span>
                                <span className="text-gray-900 dark:text-white font-mono">1.2GB / 8GB</span>
                            </div>
                            <div className="h-1.5 bg-gray-200 dark:bg-white/10 rounded-full overflow-hidden">
                                <div className="h-full bg-purple-500 w-[15%] rounded-full"></div>
                            </div>
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-500 dark:text-gray-400">Model Status</span>
                                <span className="text-green-500 font-mono flex items-center gap-1">
                                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                                    ONLINE
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
