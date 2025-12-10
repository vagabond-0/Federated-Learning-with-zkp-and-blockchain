'use client';

import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  loadCSVtoTensors,
  generateExampleData,
  normalizeTensor,
  trainModel,
  flattenWeights,
  splitWeightsForCollections,
  evaluateModel
} from '@/lib/tfutils';
import { submitModel, registerClient } from '@/lib/api';

export default function TrainPage() {
  const [file, setFile] = useState<File | null>(null);
  const [useExample, setUseExample] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<{
    epoch: number;
    loss: number;
    acc: number;
  } | null>(null);
  
  const [trainedModel, setTrainedModel] = useState<tf.LayersModel | null>(null);
  const [weights, setWeights] = useState<number[]>([]);
  const [partitions, setPartitions] = useState<{
    collectionOrg1Private: number[];
    collectionOrg2Private: number[];
  } | null>(null);
  
  const [modelID, setModelID] = useState('');
  const [clientID, setClientID] = useState('');
  const [domain, setDomain] = useState<'source' | 'target'>('source');
  const [accuracy, setAccuracy] = useState<number | null>(null);
  
  const [submitStatus, setSubmitStatus] = useState<string>('');
  const [registerStatus, setRegisterStatus] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUseExample(false);
    }
  };

  const handleUseExample = () => {
    setFile(null);
    setUseExample(true);
  };

  const handleTrain = async () => {
    try {
      setIsTraining(true);
      setTrainingProgress(null);
      setSubmitStatus('');
      
      let xs: tf.Tensor2D;
      let ys: tf.Tensor2D;
      
      // Load data
      if (useExample) {
        const data = generateExampleData(100, 8);
        xs = data.xs;
        ys = data.ys;
      } else if (file) {
        const data = await loadCSVtoTensors(file);
        xs = data.xs;
        ys = data.ys;
      } else {
        alert('Please select a file or use example data');
        setIsTraining(false);
        return;
      }
      
      // Normalize features
      const { normalized } = normalizeTensor(xs);
      
      // Train model
      const model = await trainModel(
        normalized,
        ys,
        50,
        (epoch, logs) => {
          setTrainingProgress({
            epoch: epoch + 1,
            loss: logs.loss,
            acc: logs.acc
          });
        }
      );
      
      // Evaluate model
      const evalResult = await evaluateModel(model, normalized, ys);
      setAccuracy(evalResult.accuracy);
      
      // Extract and flatten weights
      const flatWeights = flattenWeights(model);
      setWeights(flatWeights);
      
      // Split weights into partitions
      const parts = splitWeightsForCollections(flatWeights);
      setPartitions(parts);
      
      setTrainedModel(model);
      
      // Cleanup tensors
      xs.dispose();
      ys.dispose();
      normalized.dispose();
      
    } catch (error) {
      console.error('Training error:', error);
      alert(`Training failed: ${error}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleRegisterClient = async () => {
    if (!clientID || !domain) {
      alert('Please fill in clientID and domain');
      return;
    }
    
    try {
      setRegisterStatus('Registering client...');
      await registerClient(clientID, domain, 1000);
      setRegisterStatus('✅ Client registered successfully');
    } catch (error: any) {
      setRegisterStatus(`❌ Registration failed: ${error.message}`);
    }
  };

  const handleSubmit = async () => {
    if (!modelID || !clientID || !domain) {
      alert('Please fill in all required fields (modelID, clientID, domain)');
      return;
    }
    
    if (!partitions) {
      alert('Please train a model first');
      return;
    }
    
    try {
      setSubmitStatus('Submitting model to blockchain...');
      
      const response = await submitModel({
        modelID,
        clientID,
        domain,
        parts: partitions,
        meta: {
          accuracy: accuracy || 0,
          epochs: 50,
          timestamp: new Date().toISOString()
        }
      });
      
      setSubmitStatus(`✅ Model submitted successfully: ${response.message}`);
      
    } catch (error: any) {
      console.error('Submit error:', error);
      setSubmitStatus(`❌ Submit failed: ${error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          Federated Learning - Model Training
        </h1>
        
        {/* Data Selection */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 1: Select Training Data</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload CSV File
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-50 file:text-blue-700
                  hover:file:bg-blue-100"
              />
              <p className="mt-1 text-xs text-gray-500">
                CSV format: features in columns, last column is label (0 or 1)
              </p>
            </div>
            
            <div className="text-center">
              <span className="text-gray-500">or</span>
            </div>
            
            <button
              onClick={handleUseExample}
              className="w-full py-2 px-4 border border-gray-300 rounded-md
                text-sm font-medium text-gray-700 bg-white
                hover:bg-gray-50 focus:outline-none focus:ring-2
                focus:ring-offset-2 focus:ring-blue-500"
            >
              Use Example Dataset (100 samples, 8 features)
            </button>
            
            {(file || useExample) && (
              <div className="p-3 bg-green-50 rounded-md">
                <p className="text-sm text-green-800">
                  ✓ Data selected: {file ? file.name : 'Example dataset'}
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Training */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 2: Train Model</h2>
          
          <button
            onClick={handleTrain}
            disabled={isTraining || (!file && !useExample)}
            className="w-full py-3 px-4 border border-transparent rounded-md
              shadow-sm text-sm font-medium text-white bg-blue-600
              hover:bg-blue-700 focus:outline-none focus:ring-2
              focus:ring-offset-2 focus:ring-blue-500
              disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isTraining ? 'Training...' : 'Train Model (50 epochs)'}
          </button>
          
          {trainingProgress && (
            <div className="mt-4 p-4 bg-blue-50 rounded-md">
              <p className="text-sm font-medium text-blue-900">
                Training Progress
              </p>
              <div className="mt-2 space-y-1 text-sm text-blue-700">
                <p>Epoch: {trainingProgress.epoch} / 50</p>
                <p>Loss: {trainingProgress.loss.toFixed(4)}</p>
                <p>Accuracy: {(trainingProgress.acc * 100).toFixed(2)}%</p>
              </div>
            </div>
          )}
          
          {accuracy !== null && (
            <div className="mt-4 p-4 bg-green-50 rounded-md">
              <p className="text-sm font-medium text-green-900">
                ✅ Training Complete
              </p>
              <p className="text-sm text-green-700 mt-1">
                Final Accuracy: {(accuracy * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </div>
        
        {/* Weight Preview */}
        {weights.length > 0 && partitions && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Step 3: Model Weights</h2>
            
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-gray-700">
                  Total Weights: {weights.length}
                </p>
                <pre className="mt-2 p-3 bg-gray-50 rounded text-xs overflow-x-auto">
                  [{weights.slice(0, 10).map(w => w.toFixed(4)).join(', ')}
                  {weights.length > 10 ? ', ...' : ''}]
                </pre>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-700">
                    Org1 Private Collection ({partitions.collectionOrg1Private.length} weights)
                  </p>
                  <pre className="mt-2 p-3 bg-blue-50 rounded text-xs overflow-x-auto">
                    [{partitions.collectionOrg1Private.slice(0, 5).map(w => w.toFixed(4)).join(', ')}...]
                  </pre>
                </div>
                
                <div>
                  <p className="text-sm font-medium text-gray-700">
                    Org2 Private Collection ({partitions.collectionOrg2Private.length} weights)
                  </p>
                  <pre className="mt-2 p-3 bg-green-50 rounded text-xs overflow-x-auto">
                    [{partitions.collectionOrg2Private.slice(0, 5).map(w => w.toFixed(4)).join(', ')}...]
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Submit Form */}
        {partitions && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Step 4: Submit to Blockchain</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Client ID
                </label>
                <input
                  type="text"
                  value={clientID}
                  onChange={(e) => setClientID(e.target.value)}
                  placeholder="e.g., hospital-1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md
                    focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Domain
                </label>
                <select
                  value={domain}
                  onChange={(e) => setDomain(e.target.value as 'source' | 'target')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md
                    focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="source">Source (has labeled data)</option>
                  <option value="target">Target (needs predictions)</option>
                </select>
              </div>
              
              <button
                onClick={handleRegisterClient}
                className="w-full py-2 px-4 border border-blue-600 rounded-md
                  text-sm font-medium text-blue-600 bg-white
                  hover:bg-blue-50 focus:outline-none focus:ring-2
                  focus:ring-offset-2 focus:ring-blue-500"
              >
                Register Client First
              </button>
              
              {registerStatus && (
                <p className="text-sm text-gray-700">{registerStatus}</p>
              )}
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model ID
                </label>
                <input
                  type="text"
                  value={modelID}
                  onChange={(e) => setModelID(e.target.value)}
                  placeholder="e.g., model-round-1-hospital-1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md
                    focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <button
                onClick={handleSubmit}
                className="w-full py-3 px-4 border border-transparent rounded-md
                  shadow-sm text-sm font-medium text-white bg-green-600
                  hover:bg-green-700 focus:outline-none focus:ring-2
                  focus:ring-offset-2 focus:ring-green-500"
              >
                Submit Model to Blockchain
              </button>
              
              {submitStatus && (
                <div className={`p-4 rounded-md ${
                  submitStatus.includes('✅') ? 'bg-green-50' : 'bg-red-50'
                }`}>
                  <p className={`text-sm ${
                    submitStatus.includes('✅') ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {submitStatus}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}