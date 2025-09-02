import React, { useState, useEffect, useRef } from 'react';
import { Play, RefreshCw, Database, TrendingUp } from 'lucide-react';
import toast from 'react-hot-toast';
import { apiService } from '../services/api';
import IrisForm from './IrisForm';
import CancerForm from './CancerForm';
import ResultsDisplay from './ResultsDisplay';
import ModelTraining from './ModelTraining';

// 🆕 Friendly name helper ------------------------------------------
const prettyModelName = (key) => {
  switch (key) {
    case 'iris_random_forest':   return 'Iris – Random Forest';
    case 'iris_logreg':          return 'Iris – Logistic Regression';
    case 'breast_cancer_bayes':  return 'Breast Cancer – Bayesian';
    case 'breast_cancer_stub':   return 'Breast Cancer – LogReg';
    default:                     return key.replace(/_/g, ' ');
  }
};

// ---- keys we care about from backend ---------------------------------------
const STATUS_KEYS = [
  'iris_random_forest',
  'iris_logreg',
  'breast_cancer_bayes',
  'breast_cancer_stub',
];

/**
 * Sanitize backend model_status payload → {model_name: status_string}.
 * Drops large metadata entries (e.g., *_dep_audit) that crash rendering.
 */
function sanitizeModelStatus(raw) {
  if (!raw || typeof raw !== 'object') {
    console.warn('[ModelStatus] sanitize: non-object payload:', raw);
    return {};
  }

  const filtered = {};
  for (const key of STATUS_KEYS) {
    const val = raw[key];
    if (typeof val === 'string') {
      filtered[key] = val;
    } else if (val !== undefined) {
      console.warn('[ModelStatus] dropping non-string value for', key, val);
    }
  }

  // Log any unexpected keys the backend sent (debug visibility)
  for (const [k, v] of Object.entries(raw)) {
    if (!STATUS_KEYS.includes(k)) {
      console.debug('[ModelStatus] ignoring extra key from backend:', k, v);
    }
  }

  return filtered;
}

const MLModelFrontend = ({ backendReady }) => {
  const [selectedDataset, setSelectedDataset] = useState('iris');
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [modelStatus, setModelStatus] = useState({});
  const [selectedModelType, setSelectedModelType] = useState({
    iris: 'rf',
    cancer: 'bayes'
  });

  const intervalRef = useRef(null);

  useEffect(() => {
    // Only start polling once backend indicates readiness (reduces noise)
    if (!backendReady) {
      setApiStatus('loading');
      return;
    }

    checkModelStatus(); // immediate
    if (!intervalRef.current) {
      intervalRef.current = setInterval(checkModelStatus, 4000); // slower cadence
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [backendReady]);

  const checkModelStatus = async () => {
    try {
      const response = await apiService.getReadyFull();
      console.debug('[checkModelStatus] raw /ready/full response:', response);

      const filtered = sanitizeModelStatus(response.model_status);
      console.debug('[checkModelStatus] filtered model_status:', filtered);

      setModelStatus(filtered);

      if (response.ready) {
        const values = Object.values(filtered);
        const anyFailed = values.includes('failed');
        const allLoaded =
          values.length > 0 && values.every((v) => v === 'loaded');

        if (allLoaded) {
          setApiStatus('healthy');
        } else if (anyFailed) {
          setApiStatus('warning');
        } else {
          setApiStatus('loading');
        }
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      console.error('Failed to check model status:', error);
      setApiStatus('error');
    }
  };

  const handlePredict = async (formData) => {
    setIsLoading(true);
    setPredictions(null);

    try {
      let response;
      if (selectedDataset === 'iris') {
        response = await apiService.predictIris(formData);
      } else {
        response = await apiService.predictCancer(formData);
      }

      setPredictions(response);
      toast.success('Prediction completed successfully!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error(`Prediction failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    try {
      const modelType = selectedModelType[selectedDataset];
      let response;

      if (selectedDataset === 'iris') {
        response = await apiService.trainIris(modelType);
      } else if (selectedDataset === 'cancer') {
        response = await apiService.trainCancer(modelType);
      }

      toast.success(`Training started for ${selectedDataset} (${modelType})! This may take a few minutes...`);
      console.log('Training response:', response);
    } catch (error) {
      console.error('Training error:', error);
      toast.error(`Training failed: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const datasets = [
    {
      id: 'iris',
      name: 'Iris Classification',
      description: 'Classify iris flowers into species based on measurements',
      features: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
      classes: ['Setosa', 'Versicolor', 'Virginica']
    },
    {
      id: 'cancer',
      name: 'Breast Cancer Diagnosis',
      description: 'Predict malignant vs benign breast cancer diagnosis',
      features: ['30 diagnostic features'],
      classes: ['Malignant', 'Benign']
    }
  ];

  const currentDataset = datasets.find(d => d.id === selectedDataset);

  const statusColor = {
    healthy: 'bg-green-100 text-green-800',
    warning: 'bg-orange-100 text-orange-800',
    error: 'bg-red-100 text-red-800',
    loading: 'bg-yellow-100 text-yellow-800',
    checking: 'bg-gray-100 text-gray-800'
  }[apiStatus] || 'bg-gray-100 text-gray-800';

  const statusDot = {
    healthy: 'bg-green-500',
    warning: 'bg-orange-500',
    error: 'bg-red-500',
    loading: 'bg-yellow-500',
    checking: 'bg-gray-500'
  }[apiStatus] || 'bg-gray-500';

  const statusText = {
    healthy: 'All Models Ready',
    warning: 'Some Models Failed',
    error: 'API Error',
    loading: 'Models Loading…',
    checking: 'Checking…'
  }[apiStatus] || 'Checking…';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Machine Learning Models</h2>
            <p className="text-gray-600 mt-1">
              Select a dataset and make predictions using trained models
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${statusColor}`}>
              <div className={`w-2 h-2 rounded-full ${statusDot}`}></div>
              <span>{statusText}</span>
            </div>
            <button
              onClick={checkModelStatus}
              className="btn-outline btn-sm"
              disabled={apiStatus === 'checking'}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </button>
          </div>
        </div>

        {/* Model Status Details */}
        {Object.keys(modelStatus).length > 0 ? (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(modelStatus).map(([model, status]) => (
              <div
                key={model}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <span className="text-sm font-medium text-gray-700">
                  {prettyModelName(model)}
                </span>
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    status === 'loaded'
                      ? 'bg-green-100 text-green-800'
                      : status === 'training'
                      ? 'bg-blue-100 text-blue-800'
                      : status === 'failed'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {status}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-4 text-sm text-gray-500 italic">
            No model status yet…
          </div>
        )}
      </div>

      {/* Dataset Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {datasets.map((dataset) => (
          <div
            key={dataset.id}
            className={`card cursor-pointer transition-all duration-200 hover:shadow-md ${
              selectedDataset === dataset.id
                ? 'ring-2 ring-primary-500 border-primary-200'
                : 'hover:border-gray-300'
            }`}
            onClick={() => setSelectedDataset(dataset.id)}
          >
            <div className="flex items-start space-x-3">
              <div className={`p-2 rounded-lg ${
                selectedDataset === dataset.id
                  ? 'bg-primary-100 text-primary-600'
                  : 'bg-gray-100 text-gray-600'
              }`}>
                <Database className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-900">{dataset.name}</h3>
                <p className="text-sm text-gray-600 mt-1">{dataset.description}</p>
                <div className="mt-2 text-xs text-gray-500">
                  <div>Features: {dataset.features.join(', ')}</div>
                  <div className="mt-1">Classes: {dataset.classes.join(', ')}</div>
                </div>
              </div>
              {selectedDataset === dataset.id && (
                <div className="text-primary-600">
                  <TrendingUp className="h-5 w-5" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Prediction Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">
              {currentDataset?.name} Prediction
            </h3>
            <p className="text-sm text-gray-600">
              Enter values to get a prediction from the trained model
            </p>
          </div>

          {selectedDataset === 'iris' ? (
            <IrisForm 
              onPredict={handlePredict} 
              isLoading={isLoading}
              onModelTypeChange={(modelType) => setSelectedModelType(prev => ({ ...prev, iris: modelType }))}
            />
          ) : (
            <CancerForm 
              onPredict={handlePredict} 
              isLoading={isLoading}
              onModelTypeChange={(modelType) => setSelectedModelType(prev => ({ ...prev, cancer: modelType }))}
            />
          )}
        </div>

        {/* Results */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Prediction Results</h3>
            <p className="text-sm text-gray-600">
              Model predictions and confidence scores
            </p>
          </div>

          <ResultsDisplay 
            predictions={predictions} 
            dataset={selectedDataset}
            isLoading={isLoading}
          />
        </div>
      </div>

      {/* Model Training */}
      <ModelTraining 
        dataset={selectedDataset}
        onTrain={handleTrainModel}
        isTraining={isTraining}
        setIsTraining={setIsTraining}
      />
    </div>
  );
};

export default MLModelFrontend; 



