import React from 'react';
import { Zap, Clock, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';

const ModelTraining = ({ dataset, onTrain, isTraining, setIsTraining }) => {
  const getDatasetInfo = () => {
    switch (dataset) {
      case 'iris':
        return {
          name: 'Iris Classification',
          description: 'Train a new model on the Iris dataset',
          estimatedTime: '30 seconds',
          features: 4,
          samples: 150
        };
      case 'cancer':
        return {
          name: 'Breast Cancer Diagnosis',
          description: 'Train a new Bayesian model on the breast cancer dataset',
          estimatedTime: '2-3 minutes',
          features: 30,
          samples: 569
        };
      default:
        return {
          name: 'Unknown Dataset',
          description: 'Train a new model',
          estimatedTime: 'Unknown',
          features: 0,
          samples: 0
        };
    }
  };

  const pollReady = async (attempt = 0) => {
    try {
      // For now, we'll just wait a bit and then stop training
      // The parent component is already polling model status
      await new Promise(resolve => setTimeout(resolve, 2000));
      setIsTraining(false);
    } catch { /* ignore */ }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      // delegate to the parent so it can pass the right model_type
      await onTrain();
      toast.success('Training job submitted – refresh will turn green when done');
      pollReady();
    } catch (e) {
      toast.error(`Training failed: ${e.message}`);
      setIsTraining(false);
    }
  };

  const datasetInfo = getDatasetInfo();

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-medium text-gray-900">Model Training</h3>
        <p className="text-sm text-gray-600">
          Train a new model or retrain existing models with updated parameters
        </p>
      </div>

      <div className="space-y-4">
        {/* Dataset Info */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-2">{datasetInfo.name}</h4>
          <p className="text-sm text-gray-600 mb-3">{datasetInfo.description}</p>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Features:</span>
              <span className="ml-2 font-medium">{datasetInfo.features}</span>
            </div>
            <div>
              <span className="text-gray-500">Samples:</span>
              <span className="ml-2 font-medium">{datasetInfo.samples}</span>
            </div>
            <div>
              <span className="text-gray-500">Est. Time:</span>
              <span className="ml-2 font-medium">{datasetInfo.estimatedTime}</span>
            </div>
          </div>
        </div>

        {/* Training Status */}
        {isTraining && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="spinner h-5 w-5 mr-3"></div>
              <div className="flex-1">
                <h4 className="font-medium text-blue-900">Training in Progress</h4>
                <p className="text-sm text-blue-700">
                  Please wait while the model is being trained. This may take a few minutes.
                </p>
              </div>
            </div>

            <div className="mt-3">
              <div className="flex items-center text-sm text-blue-600">
                <Clock className="h-4 w-4 mr-2" />
                <span>Estimated time remaining: {datasetInfo.estimatedTime}</span>
              </div>
            </div>
          </div>
        )}

        {/* Training Options */}
        <div className="space-y-3">
          <h4 className="font-medium text-gray-900">Training Options</h4>

          {dataset === 'iris' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <h5 className="font-medium text-gray-900">Random Forest</h5>
                  <p className="text-sm text-gray-600">
                    Fast ensemble method with good performance
                  </p>
                </div>
                <div className="text-green-600">
                  <CheckCircle className="h-5 w-5" />
                </div>
              </div>

              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <h5 className="font-medium text-gray-900">Logistic Regression</h5>
                  <p className="text-sm text-gray-600">
                    Simple linear model with interpretable results
                  </p>
                </div>
                <div className="text-green-600">
                  <CheckCircle className="h-5 w-5" />
                </div>
              </div>
            </div>
          )}

          {dataset === 'cancer' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <h5 className="font-medium text-gray-900">Bayesian Model</h5>
                  <p className="text-sm text-gray-600">
                    Hierarchical Bayesian model with uncertainty quantification
                  </p>
                </div>
                <div className="text-green-600">
                  <CheckCircle className="h-5 w-5" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Training Button */}
        <div className="flex justify-center pt-4">
          <button
            onClick={handleTrain}
            disabled={isTraining}
            className="btn-primary btn-lg"
          >
            {isTraining ? (
              <>
                <div className="spinner h-5 w-5 mr-2"></div>
                Training Model...
              </>
            ) : (
              <>
                <Zap className="h-5 w-5 mr-2" />
                Train Model
              </>
            )}
          </button>
        </div>

        {/* Training Notes */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <h4 className="font-medium text-yellow-800 mb-2">Training Notes</h4>
          <ul className="text-sm text-yellow-700 space-y-1">
            <li>• Models are automatically saved after training</li>
            <li>• Previous models will be backed up before retraining</li>
            <li>• Training progress is logged in the console</li>
            {dataset === 'cancer' && (
              <li>• Bayesian models provide uncertainty estimates</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining; 


