import React, { useRef, useState, useEffect } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { Play, Pause, Activity, CheckCircle, FileText, Download, Clock } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const Player = ({ currentSong, onAnalyze, analysisResult, isAnalyzing }) => {
    const containerRef = useRef(null);
    const wavesurfer = useRef(null);
    const wsRegions = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const { theme } = useTheme();

    // Initialize Wavesurfer
    useEffect(() => {
        if (!containerRef.current) return;
        
        // Wait for container to have dimensions
        if (containerRef.current.clientWidth === 0) return;

        // Destroy previous instance
        if (wavesurfer.current) {
            wavesurfer.current.destroy();
        }

        const isDark = theme === 'dark';

        try {
            wavesurfer.current = WaveSurfer.create({
                container: containerRef.current,
                waveColor: isDark ? '#333' : '#cbd5e1',
                progressColor: isDark ? '#00f0ff' : '#2563eb',
                cursorColor: isDark ? '#fff' : '#0f172a',
                barWidth: 2,
                barGap: 3,
                barRadius: 2,
                height: 200,
                normalize: true,
                backend: 'MediaElement',
            });

            // Add Regions Plugin
            const wsRegionsPlugin = RegionsPlugin.create();
            wavesurfer.current.registerPlugin(wsRegionsPlugin);
            wsRegions.current = wsRegionsPlugin;

            // Load Audio - Use relative path or proxied path
            if (currentSong) {
                const audioUrl = `/api/audio/${currentSong.filename}`;
                // Add a small delay to ensure previous instance is fully cleaned up
                setTimeout(() => {
                    if (wavesurfer.current) {
                        wavesurfer.current.load(audioUrl).catch(err => {
                            if (err.name !== 'AbortError') {
                                console.error("Wavesurfer load error:", err);
                            }
                        });
                    }
                }, 10);
            }

            // Events
            wavesurfer.current.on('ready', () => {
                setDuration(wavesurfer.current.getDuration());
            });

            wavesurfer.current.on('audioprocess', () => {
                setCurrentTime(wavesurfer.current.getCurrentTime());
            });

            wavesurfer.current.on('finish', () => setIsPlaying(false));
            
            wavesurfer.current.on('error', (err) => {
                if (err.name === 'AbortError' || err.message?.includes('aborted')) return;
                console.error("Wavesurfer error:", err);
            });

        } catch (e) {
            console.error("Error initializing Wavesurfer:", e);
        }

        return () => {
            if (wavesurfer.current) wavesurfer.current.destroy();
        };
    }, [currentSong, theme]); // Re-init when theme changes

    // Handle Play/Pause
    const togglePlay = () => {
        if (wavesurfer.current) {
            wavesurfer.current.playPause();
            setIsPlaying(!isPlaying);
        }
    };

    // Render Regions when analysis result is available
    useEffect(() => {
        if (!wsRegions.current || !analysisResult || !wavesurfer.current) return;

        wsRegions.current.clearRegions();

        // Label Colors (Adjust opacity for light mode visibility)
        const isDark = theme === 'dark';
        const alpha = isDark ? 0.2 : 0.3;
        
        const colors = {
            'intro': `rgba(255, 99, 132, ${alpha})`,
            'verse': `rgba(54, 162, 235, ${alpha})`,
            'chorus': `rgba(255, 206, 86, ${alpha})`,
            'bridge': `rgba(75, 192, 192, ${alpha})`,
            'outro': `rgba(153, 102, 255, ${alpha})`,
            'instrumental': `rgba(255, 159, 64, ${alpha})`,
            'solo': `rgba(201, 203, 207, ${alpha})`,
            'unknown': `rgba(100, 100, 100, ${alpha})`
        };

        analysisResult.segments.forEach(seg => {
            wsRegions.current.addRegion({
                start: seg.start,
                end: seg.end,
                content: seg.label.toUpperCase(),
                color: colors[seg.label] || colors['unknown'],
                drag: false,
                resize: false
            });
        });

    }, [analysisResult, theme]);

    if (!currentSong) {
        return <div className="flex-1 flex items-center justify-center text-gray-400 dark:text-cyber-dim font-mono bg-gray-50 dark:bg-cyber-black transition-colors">请从左侧选择一首歌曲</div>;
    }

    const formatTime = (time) => {
        const min = Math.floor(time / 60);
        const sec = Math.floor(time % 60);
        return `${min}:${sec.toString().padStart(2, '0')}`;
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-gray-50 dark:bg-cyber-black relative overflow-hidden transition-colors duration-300">
            <div className="absolute inset-0 noise-overlay opacity-50 pointer-events-none"></div>
            
            {/* Top Bar */}
            <div className="h-20 border-b border-gray-200 dark:border-white/5 flex items-center justify-between px-8 bg-white/80 dark:bg-cyber-black/80 backdrop-blur-md z-10 sticky top-0">
                <div>
                    <h1 className="font-display text-4xl font-bold text-gray-900 dark:text-white mb-2">{currentSong.name}</h1>
                    <p className="font-mono text-xs text-blue-600 dark:text-cyber-primary tracking-widest">{currentSong.filename}</p>
                </div>
                <div className="flex gap-4">
                    <button 
                        onClick={onAnalyze}
                        disabled={isAnalyzing}
                        className="px-6 py-2 bg-blue-600 dark:bg-cyber-primary text-white dark:text-black font-bold hover:bg-blue-700 dark:hover:bg-cyber-secondary transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed rounded-md dark:rounded-none shadow-md dark:shadow-none"
                    >
                        {isAnalyzing ? (
                            <Activity className="w-4 h-4 animate-spin" />
                        ) : (
                            <Activity className="w-4 h-4" />
                        )}
                        {isAnalyzing ? '分析中...' : '开始分析'}
                    </button>
                </div>
            </div>

            {/* Scrollable Content Area */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8 relative z-0">
                
                {/* Visualizer Card */}
                <div className="bg-white dark:bg-black/40 border border-gray-200 dark:border-white/5 p-6 rounded-lg backdrop-blur-md shadow-xl dark:shadow-2xl">
                    <div ref={containerRef} className="w-full mb-4"></div>
                    
                    {/* Controls */}
                    <div className="flex items-center justify-between mt-4">
                        <div className="flex items-center gap-4">
                            <button 
                                onClick={togglePlay}
                                className="w-12 h-12 rounded-full bg-blue-600 dark:bg-white text-white dark:text-black flex items-center justify-center hover:bg-blue-700 dark:hover:bg-cyber-primary transition-colors shadow-lg"
                            >
                                {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-1" />}
                            </button>
                            <div className="font-mono text-blue-600 dark:text-cyber-primary text-lg tracking-widest">
                                {formatTime(currentTime)} / {formatTime(duration)}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Results Panel */}
                <div className="bg-white dark:bg-black/20 border border-gray-200 dark:border-white/5 rounded-lg backdrop-blur-sm shadow-inner overflow-hidden flex flex-col">
                    {/* Sticky Header inside Results Panel */}
                    <div className="flex items-center justify-between p-4 bg-gray-100 dark:bg-[#050505] border-b border-gray-200 dark:border-white/10 sticky top-0 z-20">
                        <div className="flex items-center gap-4">
                            <h3 className="font-display text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-blue-500 dark:text-cyber-secondary" />
                                分析结果
                            </h3>
                            {analysisResult && analysisResult.inference_time !== undefined && (
                                <span className="flex items-center gap-1.5 text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-200 dark:bg-white/5 px-2 py-1 rounded">
                                    <Clock className="w-3 h-3" />
                                    {analysisResult.inference_time.toFixed(3)}s
                                </span>
                            )}
                        </div>
                        
                        {analysisResult && (
                            <div className="flex gap-2">
                                <button 
                                    onClick={() => {
                                        const jsonContent = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(analysisResult, null, 2));
                                        const link = document.createElement("a");
                                        link.href = jsonContent;
                                        link.download = `${currentSong.filename.split('.')[0]}_structure.json`;
                                        link.click();
                                    }}
                                    className="px-3 py-1.5 border border-gray-300 dark:border-white/20 hover:border-blue-500 dark:hover:border-cyber-primary text-gray-600 dark:text-white hover:text-blue-600 dark:hover:text-cyber-primary transition-colors flex items-center gap-2 font-mono text-xs tracking-wider rounded-md dark:rounded-none"
                                >
                                    <FileText className="w-3 h-3" /> JSON
                                </button>
                                <button 
                                    onClick={() => {
                                        let csvContent = "data:text/csv;charset=utf-8,Start,End,Label\n";
                                        analysisResult.segments.forEach(seg => {
                                            csvContent += `${seg.start},${seg.end},${seg.label}\n`;
                                        });
                                        const link = document.createElement("a");
                                        link.href = encodeURI(csvContent);
                                        link.download = `${currentSong.filename.split('.')[0]}_structure.csv`;
                                        link.click();
                                    }}
                                    className="px-3 py-1.5 border border-gray-300 dark:border-white/20 hover:border-blue-500 dark:hover:border-cyber-primary text-gray-600 dark:text-white hover:text-blue-600 dark:hover:text-cyber-primary transition-colors flex items-center gap-2 font-mono text-xs tracking-wider rounded-md dark:rounded-none"
                                >
                                    <Download className="w-3 h-3" /> CSV
                                </button>
                            </div>
                        )}
                    </div>
                    
                    <div className="p-4 max-h-96 overflow-y-auto custom-scrollbar bg-gray-50 dark:bg-transparent">
                        {!analysisResult ? (
                            <div className="text-center text-gray-500 dark:text-gray-500 py-12 font-mono text-sm">
                                等待分析... 点击上方 "开始分析" 按钮
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {analysisResult.segments.map((seg, idx) => (
                                    <div key={idx} className="bg-white dark:bg-white/5 border border-gray-200 dark:border-white/5 p-3 rounded hover:border-blue-300 dark:hover:border-cyber-primary/50 transition-colors group shadow-sm dark:shadow-none">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className={`font-display font-bold text-sm px-2 py-0.5 rounded ${
                                                seg.label === 'unknown' ? 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300' : 'bg-blue-100 dark:bg-cyber-primary/20 text-blue-700 dark:text-cyber-primary'
                                            }`}>
                                                {seg.label.toUpperCase()}
                                            </span>
                                            <span className="font-mono text-xs text-gray-400 dark:text-gray-500">#{idx + 1}</span>
                                        </div>
                                        <div className="flex justify-between text-xs font-mono text-gray-500 dark:text-gray-400">
                                            <span>{seg.start.toFixed(2)}s</span>
                                            <span className="w-full border-b border-dashed border-gray-300 dark:border-gray-700 mx-2 relative top-2"></span>
                                            <span>{seg.end.toFixed(2)}s</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
            
            {/* Loading Overlay */}
            {isAnalyzing && (
                <div className="absolute inset-0 bg-white/80 dark:bg-black/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center transition-colors duration-300">
                    <div className="w-64 h-1.5 bg-gray-200 dark:bg-cyber-dim rounded-full overflow-hidden mb-4 relative">
                        {/* Use a standard indeterminate progress bar */}
                        <div className="absolute inset-0 bg-blue-600 dark:bg-cyber-primary animate-progress-indeterminate origin-left"></div>
                    </div>
                    <div className="font-mono text-blue-600 dark:text-cyber-primary animate-pulse text-sm font-bold">正在进行神经网络推理...</div>
                </div>
            )}
        </div>
    );
};

export default Player;
