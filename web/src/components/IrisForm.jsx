import React, { useState } from 'react';
import { Play, RotateCcw } from 'lucide-react';

const IrisForm = ({ onPredict, isLoading, onModelTypeChange }) => {
  const [formData, setFormData] = useState({
    model_type: 'rf',
    samples: [{
      sepal_length: 5.1,
      sepal_width: 3.5,
      petal_length: 1.4,
      petal_width: 0.2
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
      model_type: 'rf',
      samples: [{
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2
      }]
    });
  };

  const sample = formData.samples[0];

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
          <option value="rf">Random Forest</option>
          <option value="logreg">Logistic Regression</option>
        </select>
      </div>

      {/* Feature Inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="form-group">
          <label className="form-label">Sepal Length (cm)</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="10"
            value={sample.sepal_length}
            onChange={(e) => handleInputChange('sepal_length', e.target.value)}
            className="form-input"
            placeholder="e.g., 5.1"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Sepal Width (cm)</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="10"
            value={sample.sepal_width}
            onChange={(e) => handleInputChange('sepal_width', e.target.value)}
            className="form-input"
            placeholder="e.g., 3.5"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Petal Length (cm)</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="10"
            value={sample.petal_length}
            onChange={(e) => handleInputChange('petal_length', e.target.value)}
            className="form-input"
            placeholder="e.g., 1.4"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Petal Width (cm)</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="10"
            value={sample.petal_width}
            onChange={(e) => handleInputChange('petal_width', e.target.value)}
            className="form-input"
            placeholder="e.g., 0.2"
          />
        </div>
      </div>

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
              Predict Species
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
        <h4 className="text-sm font-medium text-gray-700 mb-2">Sample Data Examples:</h4>
        <div className="text-xs text-gray-600 space-y-1">
          <div><strong>Setosa:</strong> SL: 5.1, SW: 3.5, PL: 1.4, PW: 0.2</div>
          <div><strong>Versicolor:</strong> SL: 7.0, SW: 3.2, PL: 4.7, PW: 1.4</div>
          <div><strong>Virginica:</strong> SL: 6.3, SW: 3.3, PL: 6.0, PW: 2.5</div>
        </div>
      </div>
    </form>
  );
};

export default IrisForm; 
