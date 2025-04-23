#!/usr/bin/env python3
from pyannote.audio import Pipeline
import inspect
import sys

print("PyAnnote Diagnostic Tool")
print("========================")

# Load the pipeline
print("Loading pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="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"
)

# Print pipeline class and version info
print(f"\nPipeline class: {pipeline.__class__.__name__}")
print(f"Pipeline module: {pipeline.__class__.__module__}")

# Try to get pipeline version
try:
    import pyannote.audio
    print(f"pyannote.audio version: {pyannote.audio.__version__}")
except (ImportError, AttributeError):
    print("Could not determine pyannote.audio version")

# Inspect the apply method
print("\nInspecting pipeline.__call__ method:")
try:
    signature = inspect.signature(pipeline.__call__)
    print(f"Method signature: {signature}")
    print("\nParameters:")
    for name, param in signature.parameters.items():
        print(f"  {name}: {param.annotation}")
except Exception as e:
    print(f"Error inspecting __call__ method: {e}")

# Inspect the pipeline object
print("\nPipeline attributes:")
for attr in dir(pipeline):
    if not attr.startswith('_'):  # Skip private attributes
        try:
            value = getattr(pipeline, attr)
            if not callable(value):
                print(f"  {attr}: {type(value)}")
        except Exception:
            print(f"  {attr}: <error getting value>")

# Try to access instantiate method
print("\nChecking instantiate method:")
if hasattr(pipeline, 'instantiate'):
    print("  Pipeline has instantiate method")
    try:
        signature = inspect.signature(pipeline.instantiate)
        print(f"  Method signature: {signature}")
    except Exception as e:
        print(f"  Error inspecting instantiate method: {e}")
else:
    print("  Pipeline does not have instantiate method")

# Try to access parameters method
print("\nChecking parameters method:")
if hasattr(pipeline, 'parameters'):
    print("  Pipeline has parameters method")
    try:
        params = pipeline.parameters()
        print(f"  Default parameters: {params}")
    except Exception as e:
        print(f"  Error getting parameters: {e}")
else:
    print("  Pipeline does not have parameters method")

print("\nDone!")
