#!/usr/bin/env python3
"""
MLflow Model Promotion CLI

Promotes models to staging or production environments by setting MLflow aliases.
Usage: python scripts/promote.py [staging|prod] [model_name] [version]

Examples:
  python scripts/promote.py staging iris_random_forest
  python scripts/promote.py prod iris_random_forest 3
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add the api directory to Python path (now in api/scripts/)
ROOT = Path(__file__).resolve().parents[1]
api_path = ROOT.resolve()
sys.path.insert(0, str(api_path))

from app.services.ml.model_service import ModelService


async def promote_model(environment: str, model_name: str, version: int = None):
    """Promote a model to the specified environment."""

    # Map environment to stage
    stage_map = {
        'staging': 'Staging',
        'prod': 'Production'
    }

    if environment not in stage_map:
        print(f"❌ Invalid environment: {environment}. Must be 'staging' or 'prod'")
        return False

    target_stage = stage_map[environment]
    alias = "prod" if environment == "prod" else "staging"

    print(f"🚀 Promoting {model_name} to {environment} environment...")

    try:
        service = ModelService()
        await service.initialize()

        print(f"Promoting {model_name} to {target_stage}...")

        result = await service.promote_model_to_stage(
            model_name=model_name,
            target_stage=target_stage,
            version=version
        )

        if result.get("promoted"):
            print(f"✅ Successfully promoted {model_name} to {environment}")
            print(f"   Version: {result.get('version', 'N/A')}")
            print(f"   Alias: @{alias}")
            return True
        else:
            print(f"❌ Failed to promote {model_name}: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Failed to promote {model_name}: {str(e)}")
        import traceback
        # Print traceback without trying to serialize the exception object
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Promote MLflow models to staging or production")
    parser.add_argument("environment", choices=["staging", "prod"], 
                       help="Target environment (staging or prod)")
    parser.add_argument("model_name", help="Name of the model to promote")
    parser.add_argument("version", nargs="?", type=int, 
                       help="Specific version to promote (optional)")

    args = parser.parse_args()

    success = asyncio.run(promote_model(
        environment=args.environment,
        model_name=args.model_name,
        version=args.version
    ))

    if success:
        print("🎉 Model promotion completed successfully!")
        sys.exit(0)
    else:
        print("💥 Model promotion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
