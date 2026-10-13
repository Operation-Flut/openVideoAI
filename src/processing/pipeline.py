"""
Pipeline chaining for openVideoAI
Executes a sequence of EnhancementPipelines.
Passes the output of one pipeline as the input to the next.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import logging
from .base import EnhancementPipeline, ProcessingAborted

logger = logging.getLogger(__name__)

class PipelineChain:
    """
    Executes a sequence of EnhancementPipelines.
    Passes the output of one pipeline as the input to the next.
    """
    def __init__(self, pipelines: List[EnhancementPipeline]):
        self.pipelines = pipelines

    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        current_input = Path(input_path)
        
        # We need a temporary directory for intermediate results
        # In a real scenario, we'd use the TEMP_PROCESSING_DIR from state
        # For now, let's assume we use the same directory as output_path
        temp_dir = Path(output_path).parent
        temp_dir.mkdir(parents=True, exist_ok=True)

        last_result = {}
        
        try:
            for i, pipeline in enumerate(self.pipelines):
                # For the last pipeline, use the final output_path
                if i == len(self.pipelines) - 1:
                    current_output = Path(output_path)
                else:
                    # Create a temporary intermediate file
                    intermediate_filename = f"tmp_{uuid.uuid4()}_{i}_{current_input.name}"
                    current_output = temp_dir / intermediate_filename
                
                logger.info(f"Executing pipeline {i+1}/{len(self.pipelines)}: {pipeline.__class__.__name__}")
                
                # Execute the pipeline
                # Note: some pipelines might need specific kwargs
                result = pipeline.process(str(current_input), str(current_output), **kwargs)
                
                # Update input for the next pipeline
                current_input = Path(result.get('output_path', current_output))
                last_result = result

            return {
                "success": True,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "pipelines_executed": [p.__class__.__name__ for p in self.pipelines],
                "last_pipeline_result": last_result
            }
            
        except ProcessingAborted as e:
            # Cleanup intermediate files
            # This is a bit complex since we don't track all intermediate files perfectly here
            # but the worker usually handles the final output cleanup.
            raise e
        except Exception as e:
            logger.exception(f"Pipeline chain failed at step {i+1}")
            raise e