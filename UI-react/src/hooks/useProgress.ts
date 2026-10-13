import { useSelector } from 'react-redux';
import { RootState } from '../store';

export const useProgress = () => {
  const progressState = useSelector((state: RootState) => state.progress);
  return progressState;
};