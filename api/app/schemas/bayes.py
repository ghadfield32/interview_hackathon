from pydantic import BaseModel, Field, validator
from typing import Optional

class BayesCancerParams(BaseModel):
    draws: int = Field(1000, ge=200, le=20_000, description="Posterior draws retained")
    tune: int = Field(1000, ge=200, le=20_000, description="Tuning (warmup) steps")
    target_accept: float = Field(0.95, ge=0.80, le=0.999, description="NUTS target acceptance")
    compute_waic: bool = Field(True, description="Attempt WAIC (may be slow)")
    compute_loo: bool = Field(False, description="Attempt LOO (slower); auto-off by default")
    max_rhat_warn: float = Field(1.01, ge=1.0, le=1.1)
    min_ess_warn: int = Field(400, ge=50, le=5000)

    @validator("tune")
    def tune_reasonable(cls, v, values):
        if "draws" in values and v < 0.2 * values["draws"]:
            # gentle warning, not rejection
            pass
        return v

    def to_kwargs(self):
        return {
            "draws": self.draws,
            "tune": self.tune,
            "target_accept": self.target_accept,
        } 
