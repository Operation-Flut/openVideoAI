import React, { useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';

export const ProgressMonitor: React.FC<{
  isVisible: boolean;
  onClose: () => void;
}> = ({ isVisible, onClose }) => {
  const { progress, currentStage, status, error } = useSelector((state: RootState) => state.progress);
  
  useEffect(() => {
    if (isVisible && status === 'idle') {
      // Could establish WebSocket connection here
    }
  }, [isVisible, status]);
  
  if (!isVisible) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-lg w-full mx-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Pipeline Processing</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
            disabled={status === 'processing'}
          >
            ×
          </button>
        </div>
        
        {status && (
          <div>
            <div className="mb-4">
              <div className="flex justify-between mb-2">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                />
              </div>
            </div>
            
            {currentStage && (
              <div className="mb-4 p-3 bg-gray-50 rounded">
                <div className="text-sm font-medium text-gray-700">Current Stage</div>
                <div className="text-lg font-semibold text-blue-600">{currentStage}</div>
              </div>
            )}
            
            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded">
                <div className="text-sm font-medium text-red-700">Error</div>
                <div className="text-red-600">{error}</div>
              </div>
            )}
          </div>
        )}
        
        <div className="flex justify-end pt-4 border-t">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
            disabled={status === 'processing'}
          >
            {status === 'completed' ? 'Close' : status === 'error' ? 'Dismiss' : 'Cancel'}
          </button>
        </div>
      </div>
    </div>
  );
};