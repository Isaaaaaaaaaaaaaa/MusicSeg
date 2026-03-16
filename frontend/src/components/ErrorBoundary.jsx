import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full w-full p-8 bg-gray-50 dark:bg-cyber-black transition-colors">
          <div className="max-w-md text-center space-y-6 animate-slide-up">
            <div className="w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center mx-auto">
              <span className="text-2xl">⚠️</span>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">页面加载异常</h2>
            <p className="text-gray-600 dark:text-gray-400 font-mono text-sm leading-relaxed">
              系统检测到前端渲染异常。这可能是由于模型推理结果格式不兼容或音频加载被中断导致的。
            </p>
            <div className="bg-red-50 dark:bg-red-900/10 p-4 rounded-md border border-red-100 dark:border-red-900/20">
              <code className="text-xs text-red-600 dark:text-red-400 font-mono break-all">
                {this.state.error?.toString()}
              </code>
            </div>
            <button 
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-blue-600 dark:bg-cyber-primary text-white dark:text-black font-bold rounded-md dark:rounded-none shadow-lg transition-all hover:scale-105"
            >
              重新加载系统
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
