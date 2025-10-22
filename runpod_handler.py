import runpod
import json
import base64
import io
from app_runpod import app # We import this to get the app context

def handler(event):
    """
    RunPod serverless handler wrapper
    Receives event = {"input": {...}} from RunPod
    Calls our Flask app and returns results
    """
    try:
        input_data = event.get("input", {})
        endpoint = input_data.get("endpoint", "health")
        
        # Route to appropriate endpoint
        if endpoint == "health":
            # This is a simple check, doesn't need the full app
            from app_runpod import CUDA_AVAILABLE
            return {"status": "healthy", "cuda_available": CUDA_AVAILABLE}
        
        elif endpoint == "preprocess":
            # Import the function directly
            from app_runpod import preprocess_image
            
            # Extract the data needed by the function
            try:
                header, encoded = input_data['imageData'].split(",", 1)
                image_bytes = base64.b64decode(encoded)
                num_nails = int(input_data['num_nails'])
                image_resolution = int(input_data['image_resolution'])
                
                # Call the function directly and return its dictionary
                result = preprocess_image(image_bytes, num_nails, image_resolution)
                return result
            except Exception as e:
                return {"status": "error", "message": f"Error in preprocess handler: {str(e)}"}

        elif endpoint == "generate":
            # Import the function directly
            from app_runpod import generate_pattern
            
            # Call it with the full input_data dictionary
            # and return the resulting dictionary
            result = generate_pattern(input_data) 
            return result
        
        else:
            return {"status": "error", "message": f"Unknown endpoint: {endpoint}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})