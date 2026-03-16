import React, { useState, useEffect } from 'react';
import LandingPage from './components/LandingPage';
import Sidebar from './components/Sidebar';
import Player from './components/Player';
import Dashboard from './components/Dashboard';
import Library from './components/Library';
import Settings from './components/Settings';
import HistoryPage from './components/HistoryPage';
import ModelIntroPage from './components/ModelIntroPage';
import ErrorBoundary from './components/ErrorBoundary';
import { ThemeProvider } from './context/ThemeContext';

const AppContent = () => {
    const [view, setView] = useState('landing'); // 'landing' | 'dashboard' | 'library' | 'app' | 'settings' | 'intro' | 'history'
    const [songs, setSongs] = useState([]);
    const [currentSong, setCurrentSong] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isUploading, setIsUploading] = useState(false);

    // Fetch Songs
    const fetchSongs = () => {
        fetch('/api/songs')
            .then(res => res.json())
            .then(data => {
                setSongs(data.songs);
                // Don't auto-select first song anymore, let user choose from Library
            })
            .catch(err => console.error("Failed to fetch songs:", err));
    };

    useEffect(() => {
        fetchSongs();
    }, []);

    // Handle Upload
    const handleUpload = async (file) => {
        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            if (!res.ok) throw new Error('Upload failed');
            // Refresh song list
            fetchSongs();
            // Optional: Go to library after upload
            setView('library'); 
        } catch (err) {
            console.error("Upload error:", err);
            alert("上传失败，请重试。");
        } finally {
            setIsUploading(false);
        }
    };

    // Handle Analyze
    const handleAnalyze = async () => {
        if (!currentSong) return;
        setIsAnalyzing(true);
        try {
            const res = await fetch(`/api/predict/${currentSong.filename}`, {
                method: 'POST'
            });
            if (!res.ok) throw new Error('API Error');
            const data = await res.json();
            setAnalysisResult(data);
        } catch (err) {
            console.error("Analysis failed:", err);
            alert("分析失败，请检查控制台或后端日志。");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleSelectSong = (song) => {
        setCurrentSong(song);
        setAnalysisResult(null); // Clear previous result
        setView('app'); // Navigate to player
    };

    if (view === 'landing') {
        return <LandingPage onEnterApp={() => setView('dashboard')} onEnterIntro={() => setView('intro')} />;
    }

    if (view === 'intro') {
        return <ModelIntroPage onBack={() => setView('dashboard')} />;
    }

    return (
        <div className="flex h-screen w-full bg-gray-50 dark:bg-cyber-black text-gray-900 dark:text-white font-sans overflow-hidden transition-colors duration-300">
            <Sidebar 
                songs={songs} 
                currentSong={currentSong} 
                onSelectSong={handleSelectSong}
                onUpload={handleUpload}
                isUploading={isUploading}
                onNavigate={setView}
                currentView={view}
            />
            
            <main className="flex-1 flex flex-col h-full overflow-hidden bg-gray-50 dark:bg-cyber-black relative">
                <div className="absolute inset-0 noise-overlay opacity-50 pointer-events-none z-0"></div>
                
                <div className="relative z-10 flex-1 flex flex-col h-full">
                    {view === 'dashboard' && <Dashboard songs={songs} />}
                    {view === 'library' && <Library songs={songs} onSelectSong={handleSelectSong} />}
                    {view === 'settings' && <Settings />}
                    {view === 'history' && <HistoryPage />}
                    {view === 'app' && (
                        <Player 
                            currentSong={currentSong}
                            onAnalyze={handleAnalyze}
                            analysisResult={analysisResult}
                            isAnalyzing={isAnalyzing}
                        />
                    )}
                </div>
            </main>
        </div>
    );
};

const App = () => {
    return (
        <ErrorBoundary>
            <ThemeProvider>
                <AppContent />
            </ThemeProvider>
        </ErrorBoundary>
    );
};

export default App;
