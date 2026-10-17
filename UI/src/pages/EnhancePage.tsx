import { useState } from 'react'
import { MediaCanvas } from '../components/MediaCanvas'
import { usePipeline } from '../hooks/usePipeline'

export function EnhancePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const { setSelectedFiles } = usePipeline()

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file)
    // Sync to pipeline store so Export button can access it
    setSelectedFiles(file ? [file] : null)
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <MediaCanvas
        selectedFile={selectedFile}
        onFileSelect={handleFileSelect}
      />
    </div>
  )
}