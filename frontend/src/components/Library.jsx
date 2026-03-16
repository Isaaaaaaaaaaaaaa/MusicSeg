import React, { useState } from 'react';
import { Search, Music, Play, MoreVertical, Filter, Grid, List, Clock, Calendar } from 'lucide-react';

const Library = ({ songs, onSelectSong }) => {
    const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'list'
    const [searchQuery, setSearchQuery] = useState('');

    const filteredSongs = songs.filter(song => 
        song.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
        song.filename.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8 animate-slide-up">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="font-display text-3xl font-bold text-gray-900 dark:text-white">媒体库</h1>
                    <p className="text-gray-500 dark:text-gray-400 font-mono text-sm mt-1">管理您的音频素材</p>
                </div>
                
                <div className="flex items-center gap-4">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input 
                            type="text" 
                            placeholder="搜索歌曲..." 
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="pl-10 pr-4 py-2 bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-lg focus:outline-none focus:border-blue-500 dark:focus:border-cyber-primary text-sm text-gray-900 dark:text-white w-64 transition-all"
                        />
                    </div>
                    
                    <div className="flex bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-lg p-1">
                        <button 
                            onClick={() => setViewMode('grid')}
                            className={`p-2 rounded-md transition-colors ${viewMode === 'grid' ? 'bg-gray-100 dark:bg-white/10 text-blue-600 dark:text-white' : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-200'}`}
                        >
                            <Grid className="w-4 h-4" />
                        </button>
                        <button 
                            onClick={() => setViewMode('list')}
                            className={`p-2 rounded-md transition-colors ${viewMode === 'list' ? 'bg-gray-100 dark:bg-white/10 text-blue-600 dark:text-white' : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-200'}`}
                        >
                            <List className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>

            {viewMode === 'grid' ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {filteredSongs.map((song, idx) => (
                        <div key={idx} className="group relative bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-2xl overflow-hidden hover:border-blue-400 dark:hover:border-cyber-primary/50 transition-all hover:shadow-xl hover:-translate-y-1">
                            <div className="aspect-square bg-gradient-to-br from-gray-100 to-gray-200 dark:from-white/5 dark:to-white/10 flex items-center justify-center relative overflow-hidden">
                                <Music className="w-12 h-12 text-gray-300 dark:text-white/20 group-hover:scale-110 transition-transform duration-500" />
                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                                    <button 
                                        onClick={() => onSelectSong(song)}
                                        className="w-12 h-12 rounded-full bg-blue-600 dark:bg-cyber-primary text-white dark:text-black flex items-center justify-center hover:scale-110 transition-transform shadow-lg"
                                    >
                                        <Play className="w-5 h-5 ml-1 fill-current" />
                                    </button>
                                </div>
                            </div>
                            <div className="p-4">
                                <h3 className="font-display font-bold text-gray-900 dark:text-white truncate mb-1">{song.name}</h3>
                                <p className="text-xs font-mono text-gray-500 dark:text-gray-400 truncate">{song.filename}</p>
                                <div className="mt-4 flex items-center justify-between">
                                    <span className="text-xs font-mono px-2 py-1 rounded bg-gray-100 dark:bg-white/10 text-gray-500 dark:text-gray-300">
                                        MP3
                                    </span>
                                    <button className="text-gray-400 hover:text-gray-600 dark:hover:text-white transition-colors">
                                        <MoreVertical className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-2xl overflow-hidden shadow-sm">
                    <table className="w-full text-left text-sm">
                        <thead className="bg-gray-50 dark:bg-white/5 border-b border-gray-200 dark:border-white/10 text-gray-500 dark:text-gray-400 font-mono uppercase text-xs">
                            <tr>
                                <th className="px-6 py-4 font-medium">#</th>
                                <th className="px-6 py-4 font-medium">标题</th>
                                <th className="px-6 py-4 font-medium">文件名</th>
                                <th className="px-6 py-4 font-medium">添加时间</th>
                                <th className="px-6 py-4 font-medium text-right">操作</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-white/5">
                            {filteredSongs.map((song, idx) => (
                                <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group">
                                    <td className="px-6 py-4 text-gray-400 dark:text-gray-500 font-mono">{idx + 1}</td>
                                    <td className="px-6 py-4 font-medium text-gray-900 dark:text-white">{song.name}</td>
                                    <td className="px-6 py-4 text-gray-500 dark:text-gray-400 font-mono">{song.filename}</td>
                                    <td className="px-6 py-4 text-gray-500 dark:text-gray-400 font-mono text-xs">
                                        <div className="flex items-center gap-2">
                                            <Calendar className="w-3 h-3" />
                                            2024-03-15
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <button 
                                            onClick={() => onSelectSong(song)}
                                            className="px-4 py-2 bg-blue-50 dark:bg-cyber-primary/10 text-blue-600 dark:text-cyber-primary hover:bg-blue-100 dark:hover:bg-cyber-primary/20 rounded-md font-mono text-xs transition-colors font-bold tracking-wider"
                                        >
                                            分析
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default Library;
