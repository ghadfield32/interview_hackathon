import React from 'react';
import { CheckCircle, XCircle, AlertTriangle, BarChart3 } from 'lucide-react';

const ResultsDisplay = ({ predictions, dataset, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="spinner h-8 w-8 mx-auto mb-4"></div>
          <p className="text-gray-600">Making prediction...</p>
        </div>
      </div>
    );
  }

  if (!predictions) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center text-gray-500">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p>Run a prediction to see results</p>
        </div>
      </div>
    );
  }

  const renderIrisResults = () => {
    const prediction = predictions.predictions[0];
    const probabilities = predictions.probabilities[0];
    const classes = ['setosa', 'versicolor', 'virginica'];

    const getSpeciesColor = (species) => {
      switch (species.toLowerCase()) {
        case 'setosa': return 'text-green-600 bg-green-100';
        case 'versicolor': return 'text-blue-600 bg-blue-100';
        case 'virginica': return 'text-purple-600 bg-purple-100';
        default: return 'text-gray-600 bg-gray-100';
      }
    };

    return (
      <div className="space-y-4">
        {/* Main Prediction */}
        <div className="text-center">
          <div className={`inline-flex items-center px-4 py-2 rounded-full font-medium ${getSpeciesColor(prediction)}`}>
            <CheckCircle className="h-5 w-5 mr-2" />
            {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
          </div>
          <p className="text-sm text-gray-600 mt-2">Predicted Iris Species</p>
        </div>

        {/* Confidence Scores */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Confidence Scores</h4>
          {classes.map((cls, index) => (
            <div key={cls} className="flex items-center justify-between">
              <span className="text-sm text-gray-600 capitalize">{cls}</span>
              <div className="flex items-center space-x-2">
                <div className="w-24 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      cls === prediction.toLowerCase() ? 'bg-primary-600' : 'bg-gray-400'
                    }`}
                    style={{ width: `${(probabilities[index] * 100)}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-gray-700 w-12">
                  {(probabilities[index] * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderCancerResults = () => {
    const prediction  = predictions.predictions[0]?.toLowerCase();   // 'malignant' | 'benign'
    const probability = predictions.probabilities[0];
    const isMalignant = prediction.startsWith('m');  // works for full word or 'M'

    return (
      <div className="space-y-4">
        {/* Main Prediction */}
        <div className="text-center">
          <div className={`inline-flex items-center px-4 py-2 rounded-full font-medium ${
            isMalignant 
              ? 'text-red-600 bg-red-100' 
              : 'text-green-600 bg-green-100'
          }`}>
            {isMalignant ? (
              <XCircle className="h-5 w-5 mr-2" />
            ) : (
              <CheckCircle className="h-5 w-5 mr-2" />
            )}
            {isMalignant ? 'Malignant' : 'Benign'}
          </div>
          <p className="text-sm text-gray-600 mt-2">Predicted Diagnosis</p>
        </div>

        {/* Probability */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Confidence</h4>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">
              {isMalignant ? 'Malignancy' : 'Benign'} Probability
            </span>
            <div className="flex items-center space-x-2">
              <div className="w-24 bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    isMalignant ? 'bg-red-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${(probability * 100)}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-gray-700 w-12">
                {(probability * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Uncertainty (if available) */}
        {predictions.uncertainties && predictions.uncertainties[0] && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
            <div className="flex items-center">
              <AlertTriangle className="h-4 w-4 text-yellow-600 mr-2" />
              <span className="text-sm font-medium text-yellow-800">
                Uncertainty Information
              </span>
            </div>
            <p className="text-sm text-yellow-700 mt-1">
              Model uncertainty: {predictions.uncertainties[0].toFixed(3)}
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {dataset === 'iris' ? renderIrisResults() : renderCancerResults()}

      {/* Input Echo */}
      <div className="border-t pt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Input Values</h4>
        <div className="bg-gray-50 rounded-lg p-3">
          <pre className="text-xs text-gray-600 overflow-x-auto">
            {JSON.stringify(predictions.input_received[0], null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay; 
