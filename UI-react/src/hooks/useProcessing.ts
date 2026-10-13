import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store';
import { pipelineAPI } from '../services/api';
import { setProgress, setStatus, setError, setTaskId, resetProgress } from '../store/progressStore';

export const useProcessing = () => {
  const dispatch = useDispatch<AppDispatch>();
  const progressState = useSelector((state: RootState) => state.progress);
  
  const createPipeline = async (file: File, config: any, device?: string) => {
    try {
      dispatch(resetProgress());
      dispatch(setStatus('processing'));
      dispatch(setProgress(0));
      
      const response = await pipelineAPI.createPipeline(file, config, device);
      
      dispatch(setTaskId(response.task_id));
      dispatch(setStatus('completed'));
      
      return response;
    } catch (error) {
      dispatch(setStatus('error'));
      dispatch(setError(error instanceof Error ? error.message : 'Pipeline creation failed'));
      throw error;
    }
  };
  
  const getTaskStatus = async (taskId: string) => {
    try {
      const response = await pipelineAPI.getTaskStatus(taskId);
      return response;
    } catch (error) {
      dispatch(setError(error instanceof Error ? error.message : 'Failed to get task status'));
      throw error;
    }
  };
  
  return {
    ...progressState,
    createPipeline,
    getTaskStatus,
  };
};