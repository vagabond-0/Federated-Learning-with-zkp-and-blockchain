'use client';

import { useState, useEffect } from 'react';
import { parseQueryVector } from '@/lib/tfutils';
import {
  getModelDimensions,
  completeSecurePrediction,
  setupSecurePrediction,
  computePartialPrediction,
  reconstructPrediction,
  getPrediction
} from '@/lib/api';

interface PredictionResult {
  queryID: string;
  logit: number;
  prediction: number;
  timestamp: string;
}

interface WorkflowStep {
  name: string;
  status: 'pending' | 'loading' | 'success' | 'error';
  message?: string;
  timestamp?: string;
}

export default function PredictPage() {
  const [queryID, setQueryID] = useState('');
  const [queryInput, setQueryInput] = useState('');
  const [queryVector, setQueryVector] = useState<number[]>([]);
  
  const [requiredDimension, setRequiredDimension] = useState<number | null>(null);
  const [modelInfo, setModelInfo] = useState<{ version: number; round: number } | null>(null);
  
  const [useCompleteWorkflow, setUseCompleteWorkflow] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const [workflowSteps, setWorkflowSteps] = useState<WorkflowStep[]>([]);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    loadModelDimensions();
  }, []);

  const loadModelDimensions = async () => {
    try {
      const dims = await getModelDimensions();
      setRequiredDimension(dims.dimensions);
      setModelInfo({ version: dims.modelVersion, round: dims.round });
    } catch (error: any) {
      console.error('Failed to load model dimensions:', error);
    }
  };

  const handleParseQueryVector = () => {
    const parsed = parseQueryVector(queryInput);
    setQueryVector(parsed);
    
    if (requiredDimension && parsed.length !== requiredDimension) {
      alert(`Warning: Query vector has ${parsed.length} elements, but model requires ${requiredDimension}`);
    }
  };

  const generateExampleQuery = () => {
    if (!requiredDimension) {
      alert('Loading model dimensions...');
      return;
    }
    
    const example = Array.from({ length: requiredDimension }, () => 
      (Math.random() * 0.5 + 0.25).toFixed(3)
    );
    
    setQueryInput(example.join(', '));
    setQueryVector(example.map(parseFloat));
  };

  const updateWorkflowStep = (name: string, status: WorkflowStep['status'], message?: string) => {
    setWorkflowSteps(prev => {
      const existing = prev.find(s => s.name === name);
      if (existing) {
        return prev.map(s => 
          s.name === name 
            ? { ...s, status, message, timestamp: new Date().toISOString() }
            : s
        );
      }
      return [...prev, { name, status, message, timestamp: new Date().toISOString() }];
    });
  };

  const handleCompleteWorkflow = async () => {
    if (!queryID || queryVector.length === 0) {
      alert('Please provide Query ID and query vector');
      return;
    }
    
    try {
      setIsProcessing(true);
      setPredictionResult(null);
      setWorkflowSteps([]);
      
      updateWorkflowStep('complete', 'loading', 'Starting secure prediction workflow...');
      
      const response = await completeSecurePrediction({
        queryID,
        queryVector
      });
      
      updateWorkflowStep('complete', 'success', response.message);
      
      if (response.prediction) {
        setPredictionResult(response.prediction);
      }
      
      // Show workflow steps
      if (response.workflow) {
        response.workflow.forEach((step: string) => {
          updateWorkflowStep(step, 'success', `✓ ${step} completed`);
        });
      }
      
    } catch (error: any) {
      console.error('Complete workflow error:', error);
      updateWorkflowStep('complete', 'error', error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleStepByStep = async () => {
    if (!queryID || queryVector.length === 0) {
      alert('Please provide Query ID and query vector');
      return;
    }
    
    try {
      setIsProcessing(true);
      setPredictionResult(null);
      setWorkflowSteps([]);
      
      // Step 1: Setup
      updateWorkflowStep('setup', 'loading', 'Splitting query vector into shares...');
      const setupResponse = await setupSecurePrediction({ queryID, queryVector });
      updateWorkflowStep('setup', 'success', setupResponse.message);
      
      // Wait a bit for blockchain to propagate
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Step 2: Compute (both orgs - but backend handles it)
      updateWorkflowStep('compute', 'loading', 'Computing partial predictions...');
      const computeResponse = await computePartialPrediction(queryID);
      updateWorkflowStep('compute', 'success', computeResponse.message);
      
      // Wait for computation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Step 3: Reconstruct
      updateWorkflowStep('reconstruct', 'loading', 'Reconstructing final prediction...');
      const reconstructResponse = await reconstructPrediction(queryID);
      updateWorkflowStep('reconstruct', 'success', reconstructResponse.message);
      
      // Wait for reconstruction
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Step 4: Get result
      updateWorkflowStep('retrieve', 'loading', 'Retrieving final result...');
      const resultResponse = await getPrediction(queryID);
      updateWorkflowStep('retrieve', 'success', 'Result retrieved successfully');
      
      setPredictionResult(resultResponse as PredictionResult);
      
    } catch (error: any) {
      console.error('Step-by-step error:', error);
      updateWorkflowStep('error', 'error', error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusColor = (status: WorkflowStep['status']) => {
    switch (status) {
      case 'success': return 'text-green-600 bg-green-50';
      case 'loading': return 'text-blue-600 bg-blue-50';
      case 'error': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: WorkflowStep['status']) => {
    switch (status) {
      case 'success': return '✓';
      case 'loading': return '⟳';
      case 'error': return '✗';
      default: return '○';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          Secure Prediction (Privacy-Preserving)
        </h1>
        
        {/* Model Info */}
        {modelInfo && requiredDimension && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-blue-900 mb-2">Global Model Info</h3>
            <div className="grid grid-cols-3 gap-4 text-sm text-blue-800">
              <div>
                <span className="font-medium">Required Dimension:</span> {requiredDimension}
              </div>
              <div>
                <span className="font-medium">Version:</span> {modelInfo.version}
              </div>
              <div>
                <span className="font-medium">Round:</span> {modelInfo.round}
              </div>
            </div>
          </div>
        )}
        
        {/* Query Input */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 1: Prepare Query</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Query ID
              </label>
              <input
                type="text"
                value={queryID}
                onChange={(e) => setQueryID(e.target.value)}
                placeholder="e.g., patient-diagnosis-001"
                className="w-full px-3 py-2 border border-gray-300 rounded-md
                  focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Query Vector (comma-separated values)
              </label>
              <textarea
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder="e.g., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, ..."
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md
                  focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                Required: {requiredDimension || '?'} values
              </p>
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={handleParseQueryVector}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm
                  font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                Parse Vector
              </button>
              
              <button
                onClick={generateExampleQuery}
                disabled={!requiredDimension}
                className="px-4 py-2 border border-blue-600 rounded-md text-sm
                  font-medium text-blue-600 bg-white hover:bg-blue-50
                  disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Generate Example
              </button>
            </div>
            
            {queryVector.length > 0 && (
              <div className="p-3 bg-green-50 rounded-md">
                <p className="text-sm text-green-800">
                  ✓ Vector parsed: {queryVector.length} elements
                </p>
                <pre className="mt-2 text-xs text-green-700 overflow-x-auto">
                  [{queryVector.slice(0, 8).map(v => v.toFixed(3)).join(', ')}
                  {queryVector.length > 8 ? ', ...' : ''}]
                </pre>
              </div>
            )}
          </div>
        </div>
        
        {/* Workflow Selection */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 2: Execute Prediction</h2>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  checked={useCompleteWorkflow}
                  onChange={() => setUseCompleteWorkflow(true)}
                  className="mr-2"
                />
                <span className="text-sm font-medium">Complete Workflow (Recommended)</span>
              </label>
              
              <label className="flex items-center">
                <input
                  type="radio"
                  checked={!useCompleteWorkflow}
                  onChange={() => setUseCompleteWorkflow(false)}
                  className="mr-2"
                />
                <span className="text-sm font-medium">Step-by-Step (Manual)</span>
              </label>
            </div>
            
            <button
              onClick={useCompleteWorkflow ? handleCompleteWorkflow : handleStepByStep}
              disabled={isProcessing || !queryID || queryVector.length === 0}
              className="w-full py-3 px-4 border border-transparent rounded-md
                shadow-sm text-sm font-medium text-white bg-blue-600
                hover:bg-blue-700 focus:outline-none focus:ring-2
                focus:ring-offset-2 focus:ring-blue-500
                disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isProcessing ? 'Processing...' : 
                useCompleteWorkflow ? 'Run Complete Workflow' : 'Run Step-by-Step'}
            </button>
          </div>
        </div>
        
        {/* Workflow Progress */}
        {workflowSteps.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Workflow Progress</h2>
            
            <div className="space-y-3">
              {workflowSteps.map((step, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-md ${getStatusColor(step.status)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-xl">{getStatusIcon(step.status)}</span>
                      <div>
                        <p className="font-medium capitalize">{step.name}</p>
                        {step.message && (
                          <p className="text-sm mt-1">{step.message}</p>
                        )}
                      </div>
                    </div>
                    {step.timestamp && (
                      <span className="text-xs">
                        {new Date(step.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Prediction Result */}
        {predictionResult && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Prediction Result</h2>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-md">
                  <p className="text-sm text-gray-600">Query ID</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {predictionResult.queryID}
                  </p>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-md">
                  <p className="text-sm text-gray-600">Timestamp</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {new Date(predictionResult.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="p-4 bg-blue-50 rounded-md">
                <p className="text-sm text-blue-600">Raw Logit (before activation)</p>
                <p className="text-2xl font-bold text-blue-900">
                  {predictionResult.logit.toFixed(6)}
                </p>
              </div>
              
              <div className="p-6 bg-gradient-to-r from-green-50 to-green-100 rounded-lg border-2 border-green-200">
                <p className="text-sm text-green-600 mb-2">Final Prediction (probability)</p>
                <p className="text-4xl font-bold text-green-900">
                  {(predictionResult.prediction * 100).toFixed(2)}%
                </p>
                <p className="text-sm text-green-700 mt-2">
                  {predictionResult.prediction >= 0.5 
                    ? '✓ Positive Class (1)' 
                    : '✗ Negative Class (0)'}
                </p>
              </div>
              
              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
                <p className="text-sm font-medium text-yellow-900">🔒 Privacy Guarantee</p>
                <p className="text-xs text-yellow-800 mt-1">
                  Your query vector was split into secret shares. No single organization 
                  saw your complete query during computation.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}