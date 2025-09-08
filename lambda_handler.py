"""
AWS Lambda handler for Mira Backend FastAPI application.
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from mangum import Mangum
from app.main import app

# Create the Lambda handler
handler = Mangum(app, lifespan="off")

# Optional: Add custom middleware for Lambda-specific logging
def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    This function wraps the FastAPI app with Mangum to make it compatible
    with AWS Lambda's event/context interface.
    """
    # Add Lambda context to the event for potential use in the app
    event["lambda_context"] = {
        "function_name": context.function_name,
        "function_version": context.function_version,
        "invoked_function_arn": context.invoked_function_arn,
        "memory_limit_in_mb": context.memory_limit_in_mb,
        "remaining_time_in_millis": context.get_remaining_time_in_millis(),
    }

    return handler(event, context)
