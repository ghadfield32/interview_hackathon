from pydantic import BaseModel, Field
from typing import List, Optional

class IrisFeatures(BaseModel):
    """Iris measurement features."""
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)

class IrisPredictRequest(BaseModel):
    """Iris prediction request (accepts legacy 'rows' alias)."""
    model_type: str = Field("rf", description="Model type: 'rf' or 'logreg'")
    samples: List[IrisFeatures] = Field(
        ...,
        description="List of iris measurements",
        alias="rows",
    )

    class Config:
        populate_by_name = True
        extra = "forbid"

class IrisPredictResponse(BaseModel):
    """Iris prediction response."""
    predictions: List[str] = Field(..., description="Predicted iris species")
    probabilities: List[List[float]] = Field(..., description="Class probabilities")
    input_received: List[IrisFeatures] = Field(..., description="Echo of input features") 
