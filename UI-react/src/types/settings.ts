export interface Settings {
  computeBackend: 'auto' | 'cuda' | 'cpu' | 'metal' | 'hip' | 'amd' | 'dml';
  gpuDevice: string;
  outputFormat: 'source' | 'jpg' | 'png' | 'webp' | 'tiff' | 'mp4';
  jpgQuality: number;
  defaultTileSize: number;
  fp16Enabled: boolean;
  cpuThreads: number;
  maxVramUsage: number;
  sharpenOverlap: number;
  sharpenUnsharpAmount: number;
  modelDir: string;
  exportDir: string;
}

export const defaultSettings: Settings = {
  computeBackend: 'auto',
  gpuDevice: 'auto',
  outputFormat: 'jpg',
  jpgQuality: 95,
  defaultTileSize: 400,
  fp16Enabled: true,
  cpuThreads: 8,
  maxVramUsage: 95,
  sharpenOverlap: 128,
  sharpenUnsharpAmount: 0.5,
  modelDir: 'models',
  exportDir: 'exports',
};