import React, { useState, useCallback } from 'react';

export const FileDropZone: React.FC<{
  onFilesSelected: (files: File[]) => void;
  acceptedTypes?: string[];
  maxFiles?: number;
}> = ({ onFilesSelected, acceptedTypes = [], maxFiles = 10 }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  }, []);
  
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(Array.from(e.target.files));
    }
  }, []);
  
  const handleFiles = (files: File[]) => {
    setError(null);
    
    // Validate file types
    if (acceptedTypes.length > 0) {
      const invalidFiles = files.filter(file => 
        !acceptedTypes.some(type => file.type.startsWith(type.split('/')[0]) || file.type === type)
      );
      
      if (invalidFiles.length > 0) {
        setError(`Invalid file types: ${invalidFiles.map(f => f.name).join(', ')}`);
        return;
      }
    }
    
    // Check max files
    if (selectedFiles.length + files.length > maxFiles) {
      setError(`Maximum ${maxFiles} files allowed`);
      return;
    }
    
    const newFiles = [...selectedFiles, ...files].slice(0, maxFiles);
    setSelectedFiles(newFiles);
    onFilesSelected(newFiles);
  };
  
  const removeFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    setSelectedFiles(newFiles);
    onFilesSelected(newFiles);
  };
  
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  return (
    <div className="border-2 border-dashed rounded-lg p-8 bg-white">
      <input
        type="file"
        onChange={handleFileSelect}
        className="hidden"
        id="file-input"
        multiple
        accept={acceptedTypes.join(',')}
      />
      
      <label
        htmlFor="file-input"
        className={`
          cursor-pointer block p-8 border-2 border-dashed rounded-lg
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50'}
          hover:border-blue-400 hover:bg-blue-25 transition-all
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="text-center">
          <div className="text-5xl mb-4">📁</div>
          <div className="text-lg font-medium mb-2">
            {isDragging ? 'Drop files here' : 'Drag and drop files here'}
          </div>
          <div className="text-sm text-gray-600 mb-4">
            or <span className="text-blue-600 font-medium">click to select</span> files
          </div>
          
          {acceptedTypes.length > 0 && (
            <div className="text-xs text-gray-500 mb-2">
              Accepted types: {acceptedTypes.join(', ')}
            </div>
          )}
          
          {maxFiles > 1 && (
            <div className="text-xs text-gray-500">
              Maximum {maxFiles} files
            </div>
          )}
        </div>
      </label>
      
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {error}
        </div>
      )}
      
      {selectedFiles.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium">Selected Files ({selectedFiles.length}/{maxFiles})</h3>
            {selectedFiles.length > 0 && (
              <button
                onClick={() => { setSelectedFiles([]); onFilesSelected([]); }}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Clear all
              </button>
            )}
          </div>
          
          <ul className="space-y-2 max-h-60 overflow-y-auto">
            {selectedFiles.map((file, index) => (
              <li key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded border">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <span className="text-xl">{
                    file.type.startsWith('video/') ? '🎬' :
                    file.type.startsWith('image/') ? '🖼️' : '📄'
                  }</span>
                  <div className="min-w-0">
                    <div className="font-medium text-sm truncate">{file.name}</div>
                    <div className="text-xs text-gray-500">
                      {formatFileSize(file.size)} • {file.type || 'Unknown type'}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-red-500 hover:text-red-700 p-1"
                  aria-label="Remove file"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};