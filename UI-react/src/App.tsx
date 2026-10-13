import { useState } from 'react';
import { Provider } from 'react-redux';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router } from 'react-router-dom';
import { PipelineBuilder } from './components/PipelineBuilder/PipelineBuilder';
import { ProgressMonitor } from './components/ProgressMonitor/ProgressMonitor';
import { SettingsPanel } from './components/SettingsPanel/SettingsPanel';
import { FileDropZone } from './components/FileDropZone/FileDropZone';
import { store } from './store';
import { apiClient } from './services/api';

const queryClient = new QueryClient();

function App() {
  const [isProgressMonitorOpen, setProgressMonitorOpen] = useState(false);
  const [pipelineConfig, setPipelineConfig] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  const handlePipelineCreated = async (config: any, file: File) => {
    try {
      setPipelineConfig(config);
      setSelectedFile(file);
      
      const response = await apiClient.post('/processing/pipeline/file', {
        pipeline_config: config,
        file: file,
        device: 'cuda'
      });
      
      const taskId = response.data.task_id;
      setProgressMonitorOpen(true);
      
      const pollProgress = async () => {
        try {
          const statusResponse = await apiClient.get(`/processing/task/${taskId}`);
          const { progress, status, currentStage } = statusResponse.data;
          
          console.log('Progress update:', progress, currentStage, status);
          
          if (status === 'completed' || status === 'error') {
            return;
          }
        } catch (err) {
          console.error('Progress polling error:', err);
        }
      };
      
      const intervalId = setInterval(pollProgress, 1000);
      return () => clearInterval(intervalId);
    } catch (error) {
      console.error('Pipeline creation error:', error);
    }
  };
  
  const handleFileDrop = (files: File[]) => {
    if (files.length > 0) {
      setSelectedFile(files[0]);
    }
  };
  
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="min-h-screen bg-gray-100">
            <header className="bg-white shadow-sm">
              <div className="max-w-7xl mx-auto px-4 py-4">
                <h1 className="text-2xl font-bold text-gray-900">openVideoAI Pipeline Builder</h1>
                <p className="text-gray-500 mt-1">Build and execute multi-stage video/image enhancement pipelines</p>
              </div>
            </header>
            
            <main className="max-w-7xl mx-auto px-4 py-8">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-6">
                  <PipelineBuilder />
                  
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <h2 className="text-xl font-bold mb-4">Input File</h2>
                    <FileDropZone 
                      onFilesSelected={handleFileDrop}
                      acceptedTypes={['image/', 'video/']}
                      maxFiles={1}
                    />
                    {selectedFile && (
                      <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded">
                        <div className="font-medium">Selected: {selectedFile.name}</div>
                        <div className="text-sm text-gray-600">
                          Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Type: {selectedFile.type}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <button
                      onClick={() => {
                        if (selectedFile && pipelineConfig) {
                          handlePipelineCreated(pipelineConfig, selectedFile);
                        }
                      }}
                      disabled={!selectedFile || !pipelineConfig}
                      className="w-full px-6 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                    >
                      {pipelineConfig ? (selectedFile ? 'Execute Pipeline' : 'Select input file') : 'Build a pipeline first'}
                    </button>
                  </div>
                </div>
                
                <div className="space-y-6">
                  <SettingsPanel />
                </div>
              </div>
            </main>
            
            <ProgressMonitor 
              isVisible={isProgressMonitorOpen}
              onClose={() => setProgressMonitorOpen(false)}
            />
          </div>
        </Router>
      </QueryClientProvider>
    </Provider>
  );
}

export default App;