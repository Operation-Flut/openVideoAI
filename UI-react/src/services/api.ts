import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Pipeline API endpoints
export const pipelineAPI = {
  // Create pipeline execution
  createPipeline: async (file: File, config: any, device?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('pipeline_config', JSON.stringify(config));
    if (device) {
      formData.append('device', device);
    }

    const response = await apiClient.post('/processing/pipeline/file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  // Get task status
  getTaskStatus: async (taskId: string) => {
    const response = await apiClient.get(`/processing/task/${taskId}`);
    return response.data;
  },
  
  // Get processing features
  getProcessingFeatures: async () => {
    const response = await apiClient.get('/processing/features');
    return response.data;
  },
  
  // Get available models for a feature
  getModelsByFeature: async (feature: string) => {
    const response = await apiClient.get(`/models?feature=${feature}`);
    return response.data;
  },
};

// Models API
export const modelsAPI = {
  getAllModels: async () => {
    const response = await apiClient.get('/models');
    return response.data;
  },
  
  getModel: async (name: string) => {
    const response = await apiClient.get(`/models/${name}`);
    return response.data;
  },
  
  downloadModel: async (modelName: string) => {
    const response = await apiClient.post('/models/download', { model_name: modelName });
    return response.data;
  },
  
  getDownloadProgress: async (modelName: string) => {
    const response = await apiClient.get(`/models/${modelName}/progress`);
    return response.data;
  },
  
  cancelDownload: async (modelName: string) => {
    const response = await apiClient.post(`/models/${modelName}/cancel`);
    return response.data;
  },
  
  deleteModel: async (modelName: string) => {
    const response = await apiClient.delete(`/models/${modelName}`);
    return response.data;
  },
};

// Settings API
export const settingsAPI = {
  getSettings: async () => {
    const response = await apiClient.get('/settings');
    return response.data;
  },
  
  updateSettings: async (settings: any) => {
    const response = await apiClient.post('/settings', settings);
    return response.data;
  },
  
  updateModelDir: async (dir: string) => {
    const response = await apiClient.post('/settings/model-dir', { path: dir });
    return response.data;
  },
  
  updateExportDir: async (dir: string) => {
    const response = await apiClient.post('/settings/export-dir', { path: dir });
    return response.data;
  },
};

// System info API
export const systemAPI = {
  getSystemInfo: async () => {
    const response = await apiClient.get('/system/info');
    return response.data;
  },
  
  getGPUs: async () => {
    const response = await apiClient.get('/system/gpus');
    return response.data;
  },
  
  getCapabilities: async () => {
    const response = await apiClient.get('/system/capabilities');
    return response.data;
  },
};

export { apiClient };