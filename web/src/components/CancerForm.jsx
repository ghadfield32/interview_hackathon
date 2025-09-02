import React, { useState } from 'react';
import { Play, RotateCcw, ChevronDown, ChevronUp } from 'lucide-react';

const CancerForm = ({ onPredict, isLoading, onModelTypeChange }) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [formData, setFormData] = useState({
    model_type: 'bayes',
    samples: [{
      // Mean features
      mean_radius: 14.13,
      mean_texture: 19.26,
      mean_perimeter: 91.97,
      mean_area: 654.89,
      mean_smoothness: 0.096,
      mean_compactness: 0.104,
      mean_concavity: 0.089,
      mean_concave_points: 0.048,
      mean_symmetry: 0.181,
      mean_fractal_dimension: 0.063,

      // SE features
      se_radius: 0.406,
      se_texture: 1.216,
      se_perimeter: 2.866,
      se_area: 40.34,
      se_smoothness: 0.007,
      se_compactness: 0.025,
      se_concavity: 0.032,
      se_concave_points: 0.012,
      se_symmetry: 0.020,
      se_fractal_dimension: 0.004,

      // Worst features
      worst_radius: 16.27,
      worst_texture: 25.68,
      worst_perimeter: 107.26,
      worst_area: 880.58,
      worst_smoothness: 0.132,
      worst_compactness: 0.254,
      worst_concavity: 0.273,
      worst_concave_points: 0.114,
      worst_symmetry: 0.290,
      worst_fractal_dimension: 0.084
    }]
  });

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      samples: [{
        ...prev.samples[0],
        [field]: parseFloat(value) || 0
      }]
    }));
  };

  const handleModelTypeChange = (value) => {
    setFormData(prev => ({
      ...prev,
      model_type: value
    }));
    if (onModelTypeChange) {
      onModelTypeChange(value);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(formData);
  };

  const handleReset = () => {
    setFormData({
      model_type: 'bayes',
      samples: [{
        mean_radius: 14.13,
        mean_texture: 19.26,
        mean_perimeter: 91.97,
        mean_area: 654.89,
        mean_smoothness: 0.096,
        mean_compactness: 0.104,
        mean_concavity: 0.089,
        mean_concave_points: 0.048,
        mean_symmetry: 0.181,
        mean_fractal_dimension: 0.063,
        se_radius: 0.406,
        se_texture: 1.216,
        se_perimeter: 2.866,
        se_area: 40.34,
        se_smoothness: 0.007,
        se_compactness: 0.025,
        se_concavity: 0.032,
        se_concave_points: 0.012,
        se_symmetry: 0.020,
        se_fractal_dimension: 0.004,
        worst_radius: 16.27,
        worst_texture: 25.68,
        worst_perimeter: 107.26,
        worst_area: 880.58,
        worst_smoothness: 0.132,
        worst_compactness: 0.254,
        worst_concavity: 0.273,
        worst_concave_points: 0.114,
        worst_symmetry: 0.290,
        worst_fractal_dimension: 0.084
      }]
    });
  };

  const sample = formData.samples[0];

  const meanFeatures = [
    { key: 'mean_radius', label: 'Mean Radius', step: 0.001, max: 30 },
    { key: 'mean_texture', label: 'Mean Texture', step: 0.001, max: 50 },
    { key: 'mean_perimeter', label: 'Mean Perimeter', step: 0.001, max: 200 },
    { key: 'mean_area', label: 'Mean Area', step: 0.001, max: 2500 },
    { key: 'mean_smoothness', label: 'Mean Smoothness', step: 0.001, max: 0.2 },
    { key: 'mean_compactness', label: 'Mean Compactness', step: 0.001, max: 0.5 },
    { key: 'mean_concavity', label: 'Mean Concavity', step: 0.001, max: 0.5 },
    { key: 'mean_concave_points', label: 'Mean Concave Points', step: 0.001, max: 0.2 },
    { key: 'mean_symmetry', label: 'Mean Symmetry', step: 0.001, max: 0.4 },
    { key: 'mean_fractal_dimension', label: 'Mean Fractal Dimension', step: 0.001, max: 0.1 }
  ];

  const seFeatures = [
    { key: 'se_radius', label: 'SE Radius', step: 0.001, max: 3 },
    { key: 'se_texture', label: 'SE Texture', step: 0.001, max: 5 },
    { key: 'se_perimeter', label: 'SE Perimeter', step: 0.001, max: 20 },
    { key: 'se_area', label: 'SE Area', step: 0.1, max: 500 },
    { key: 'se_smoothness', label: 'SE Smoothness', step: 0.0001, max: 0.05 },
    { key: 'se_compactness', label: 'SE Compactness', step: 0.001, max: 0.1 },
    { key: 'se_concavity', label: 'SE Concavity', step: 0.001, max: 0.2 },
    { key: 'se_concave_points', label: 'SE Concave Points', step: 0.001, max: 0.05 },
    { key: 'se_symmetry', label: 'SE Symmetry', step: 0.001, max: 0.1 },
    { key: 'se_fractal_dimension', label: 'SE Fractal Dimension', step: 0.0001, max: 0.02 }
  ];

  const worstFeatures = [
    { key: 'worst_radius', label: 'Worst Radius', step: 0.1, max: 40 },
    { key: 'worst_texture', label: 'Worst Texture', step: 0.1, max: 60 },
    { key: 'worst_perimeter', label: 'Worst Perimeter', step: 0.1, max: 300 },
    { key: 'worst_area', label: 'Worst Area', step: 1, max: 4000 },
    { key: 'worst_smoothness', label: 'Worst Smoothness', step: 0.001, max: 0.3 },
    { key: 'worst_compactness', label: 'Worst Compactness', step: 0.001, max: 1.0 },
    { key: 'worst_concavity', label: 'Worst Concavity', step: 0.001, max: 1.0 },
    { key: 'worst_concave_points', label: 'Worst Concave Points', step: 0.001, max: 0.3 },
    { key: 'worst_symmetry', label: 'Worst Symmetry', step: 0.001, max: 0.7 },
    { key: 'worst_fractal_dimension', label: 'Worst Fractal Dimension', step: 0.001, max: 0.2 }
  ];

  const renderFeatureGroup = (features, title) => (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-gray-700">{title}</h4>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {features.map((feature) => (
          <div key={feature.key} className="form-group">
            <label className="form-label text-xs">{feature.label}</label>
            <input
              type="number"
              step={feature.step}
              min="0"
              max={feature.max}
              value={sample[feature.key]}
              onChange={(e) => handleInputChange(feature.key, e.target.value)}
              className="form-input text-sm"
            />
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Model Type Selection */}
      <div className="form-group">
        <label className="form-label">Model Type</label>
        <select
          value={formData.model_type}
          onChange={(e) => handleModelTypeChange(e.target.value)}
          className="form-input"
        >
          <option value="bayes">Bayesian Model</option>
          <option value="stub">Logreg (Fast fallback)</option>
        </select>
      </div>

      {/* Primary Features (Mean) */}
      {renderFeatureGroup(meanFeatures.slice(0, 4), "Primary Features")}

      {/* Advanced Features Toggle */}
      <div className="border-t pt-4">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center text-sm text-primary-600 hover:text-primary-700"
        >
          {showAdvanced ? (
            <>
              <ChevronUp className="h-4 w-4 mr-1" />
              Hide Advanced Features
            </>
          ) : (
            <>
              <ChevronDown className="h-4 w-4 mr-1" />
              Show All Features (30 total)
            </>
          )}
        </button>
      </div>

      {/* Advanced Features */}
      {showAdvanced && (
        <div className="space-y-6 bg-gray-50 p-4 rounded-lg">
          {renderFeatureGroup(meanFeatures.slice(4), "Additional Mean Features")}
          {renderFeatureGroup(seFeatures, "Standard Error Features")}
          {renderFeatureGroup(worstFeatures, "Worst Features")}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex space-x-3 pt-4">
        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary flex-1"
        >
          {isLoading ? (
            <>
              <div className="spinner h-4 w-4 mr-2"></div>
              Predicting...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Predict Diagnosis
            </>
          )}
        </button>

        <button
          type="button"
          onClick={handleReset}
          className="btn-outline"
          disabled={isLoading}
        >
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset
        </button>
      </div>

      {/* Sample Data Info */}
      <div className="bg-gray-50 rounded-lg p-3 mt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">About the Features:</h4>
        <div className="text-xs text-gray-600 space-y-1">
          <div><strong>Mean:</strong> Average values of cell nucleus features</div>
          <div><strong>SE:</strong> Standard error of the features</div>
          <div><strong>Worst:</strong> Worst (largest) values of the features</div>
        </div>
      </div>
    </form>
  );
};

export default CancerForm; 
