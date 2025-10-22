import runpod
import json
from app_runpod import app
import io
from flask import Response

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
            return {"status": "healthy", "cuda_available": True}
        
        elif endpoint == "preprocess":
            # Import here to avoid circular imports
            from app_runpod import preprocess_image
            with app.test_request_context(json=input_data):
                response = preprocess_image()
                return json.loads(response[0].get_data())
        
        elif endpoint == "generate":
            from app_runpod import generate_pattern
            with app.test_request_context(json=input_data):
                response = generate_pattern()
                return json.loads(response[0].get_data())
        
        else:
            return {"status": "error", "message": f"Unknown endpoint: {endpoint}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})