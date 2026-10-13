import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store';
import { setModels, setLoading, setError } from '../store/pipelineStore';
import { modelsAPI } from '../services/api';

export const useModels = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { models, loading, error } = useSelector((state: RootState) => state.pipeline);
  
  const loadAllModels = async () => {
    dispatch(setLoading(true));
    try {
      const response = await modelsAPI.getAllModels();
      dispatch(setModels(response));
    } catch (err) {
      dispatch(setError(err instanceof Error ? err.message : 'Failed to load models'));
    }
  };
  
  return {
    models,
    loading,
    error,
    loadAllModels,
  };
};

export const useFeatureModels = (feature: string) => {
  const dispatch = useDispatch<AppDispatch>();
  const { models, loading, error } = useSelector((state: RootState) => state.pipeline);
  
  const loadFeatureModels = async () => {
    if (models[feature] === undefined) {
      dispatch(setLoading(true));
      try {
        // This would need to be implemented in the API
        const response = await modelsAPI.getAllModels();
        dispatch(setModels({ ...models, [feature]: response }));
      } catch (err) {
        dispatch(setError(err instanceof Error ? err.message : 'Failed to load models'));
      }
    }
  };
  
  return {
    models: models[feature] || [],
    loading,
    error,
    loadFeatureModels,
  };
};