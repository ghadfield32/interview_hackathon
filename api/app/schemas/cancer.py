from pydantic import BaseModel, Field
from typing import List, Optional

class CancerFeatures(BaseModel):
    """Breast cancer diagnostic features."""
    mean_radius: float = Field(..., description="Mean of distances from center to points on perimeter")
    mean_texture: float = Field(..., description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., description="Mean size of the core tumor")
    mean_area: float = Field(..., description="Mean area of the core tumor")
    mean_smoothness: float = Field(..., description="Mean of local variation in radius lengths")
    mean_compactness: float = Field(..., description="Mean of perimeter^2 / area - 1.0")
    mean_concavity: float = Field(..., description="Mean of severity of concave portions of the contour")
    mean_concave_points: float = Field(..., description="Mean for number of concave portions of the contour")
    mean_symmetry: float = Field(..., description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., description="Mean for 'coastline approximation' - 1")

    # SE features (standard error)
    se_radius: float = Field(..., description="Standard error of radius")
    se_texture: float = Field(..., description="Standard error of texture")
    se_perimeter: float = Field(..., description="Standard error of perimeter")
    se_area: float = Field(..., description="Standard error of area")
    se_smoothness: float = Field(..., description="Standard error of smoothness")
    se_compactness: float = Field(..., description="Standard error of compactness")
    se_concavity: float = Field(..., description="Standard error of concavity")
    se_concave_points: float = Field(..., description="Standard error of concave points")
    se_symmetry: float = Field(..., description="Standard error of symmetry")
    se_fractal_dimension: float = Field(..., description="Standard error of fractal dimension")

    # Worst features
    worst_radius: float = Field(..., description="Worst radius")
    worst_texture: float = Field(..., description="Worst texture")
    worst_perimeter: float = Field(..., description="Worst perimeter")
    worst_area: float = Field(..., description="Worst area")
    worst_smoothness: float = Field(..., description="Worst smoothness")
    worst_compactness: float = Field(..., description="Worst compactness")
    worst_concavity: float = Field(..., description="Worst concavity")
    worst_concave_points: float = Field(..., description="Worst concave points")
    worst_symmetry: float = Field(..., description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., description="Worst fractal dimension")

class CancerPredictRequest(BaseModel):
    """Cancer prediction request (allows 'rows' alias)."""
    model_type: str = Field("bayes", description="Model type: 'bayes', 'logreg', or 'rf'")
    samples: List[CancerFeatures] = Field(
        ...,
        description="Breast-cancer feature vectors",
        alias="rows",
    )
    posterior_samples: Optional[int] = Field(
        None, ge=10, le=10_000, description="Posterior draws for uncertainty"
    )

    class Config:
        populate_by_name = True
        extra = "forbid"

class CancerPredictResponse(BaseModel):
    """Cancer prediction response."""
    predictions: List[str] = Field(..., description="Predicted diagnosis (M=malignant, B=benign)")
    probabilities: List[float] = Field(..., description="Probability of malignancy")
    uncertainties: Optional[List[float]] = Field(None, description="Uncertainty estimates (if requested)")
    input_received: List[CancerFeatures] = Field(..., description="Echo of input features") 
