from pyannote.audio import Pipeline
import inspect
import sys

def discover_pipeline_parameters():
    """
    Discover the parameters supported by the pyannote.audio diarization pipeline
    """
    print("Loading pyannote.audio pipeline...")
    
    try:
        # Load the pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token="[TOKEN]"
        )
        
        # Get the pipeline class
        pipeline_class = pipeline.__class__
        print(f"Pipeline class: {pipeline_class.__name__}")
        
        # Try to get the apply method signature
        try:
            apply_method = getattr(pipeline_class, "apply", None)
            if apply_method:
                signature = inspect.signature(apply_method)
                print("\nParameters for pipeline.apply method:")
                for param_name, param in signature.parameters.items():
                    if param_name not in ["self", "file"]:
                        print(f"  - {param_name}: {param.default if param.default is not inspect.Parameter.empty else 'Required'}")
            else:
                print("Could not find 'apply' method")
        except Exception as e:
            print(f"Error getting method signature: {e}")
        
        # Try to get pipeline parameters from instantiation
        try:
            print("\nTrying to get pipeline parameters from instantiation...")
            
            # Check if there's a default parameters attribute
            if hasattr(pipeline, "parameters_"):
                print("\nDefault parameters from pipeline.parameters_:")
                for param_name, param_value in pipeline.parameters_.items():
                    print(f"  - {param_name}: {param_value}")
            
            # Check if there's a get_parameters method
            if hasattr(pipeline, "get_parameters"):
                print("\nParameters from pipeline.get_parameters():")
                params = pipeline.get_parameters()
                for param_name, param_value in params.items():
                    print(f"  - {param_name}: {param_value}")
                    
        except Exception as e:
            print(f"Error getting pipeline parameters: {e}")
        
        # Try to run with a test parameter to see if it's accepted
        test_params = [
            "num_speakers",
            "min_speakers", 
            "max_speakers",
            "segmentation",
            "clustering",
            "embedding",
            "clustering_method",
            "clustering_threshold",
            "segmentation_threshold"
        ]
        
        print("\nTesting which parameters are accepted:")
        for param in test_params:
            try:
                # Create a test dictionary with just this parameter
                if param == "segmentation":
                    test_dict = {param: {"min_duration_off": 0.5}}
                elif param == "embedding":
                    test_dict = {param: {"batch_size": 32}}
                else:
                    test_dict = {param: 2}
                
                # Try to validate the parameter
                if hasattr(pipeline, "check_parameters"):
                    pipeline.check_parameters(test_dict)
                    print(f"  - {param}: ACCEPTED")
                else:
                    # If no validation method, just see if the attribute exists
                    if hasattr(pipeline, param) or param in getattr(pipeline, "parameters_", {}):
                        print(f"  - {param}: LIKELY ACCEPTED (attribute exists)")
                    else:
                        print(f"  - {param}: UNKNOWN (no validation method)")
            except Exception as e:
                print(f"  - {param}: REJECTED ({str(e)})")
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")

if __name__ == "__main__":
    discover_pipeline_parameters()
