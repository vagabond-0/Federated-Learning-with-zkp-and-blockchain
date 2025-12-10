const API_BASE_URL = process.env.NEXT_PUBLIC_FLASK_API_BASE_URL || 'http://localhost:5000';

/**
 * API response types
 */
export interface ApiResponse {
  message?: string;
  error?: string;
  [key: string]: any;
}

export interface ModelSubmitRequest {
  modelID: string;
  clientID: string;
  domain: 'source' | 'target';
  parts: {
    collectionOrg1Private: number[];
    collectionOrg2Private: number[];
  };
  meta?: {
    accuracy?: number;
    epochs?: number;
    timestamp?: string;
    [key: string]: any;
  };
}

export interface PredictionRequest {
  queryID: string;
  queryVector: number[];
}

export interface ClientRegisterRequest {
  clientID: string;
  domain: 'source' | 'target';
  datasetSize: number;
}

export interface AggregationRequest {
  modelClientPairs: string[]; // Format: ["modelID::clientID", ...]
  beta?: number; // 0 or 1
}

/**
 * Register a new client
 */
export async function registerClient(
  clientID: string,
  domain: 'source' | 'target',
  datasetSize: number
): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/client/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      clientID,
      domain,
      datasetSize,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to register client');
  }

  return response.json();
}

/**
 * Check if client exists
 */
export async function checkClientExists(clientID: string): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/api/client/${clientID}/exists`);
  const data = await response.json();
  return data.exists;
}

/**
 * Get client details
 */
export async function getClient(clientID: string): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/client/${clientID}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get client');
  }
  
  return response.json();
}

/**
 * Submit local model with private data partitions
 */
export async function submitModel(request: ModelSubmitRequest): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/model/submit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to submit model');
  }

  return response.json();
}

/**
 * Aggregate models using VPSA
 */
export async function aggregateModelsVPSA(request: AggregationRequest): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/aggregate/vpsa`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to aggregate models');
  }

  return response.json();
}

/**
 * Get global model
 */
export async function getGlobalModel(): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/global-model`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get global model');
  }
  
  return response.json();
}

/**
 * Get model dimensions
 */
export async function getModelDimensions(): Promise<{
  dimensions: number;
  modelVersion: number;
  round: number;
}> {
  const response = await fetch(`${API_BASE_URL}/api/model/dimensions`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get model dimensions');
  }
  
  return response.json();
}

/**
 * Complete secure prediction workflow
 */
export async function completeSecurePrediction(request: PredictionRequest): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/prediction/complete`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to complete prediction');
  }

  return response.json();
}

/**
 * Setup secure prediction (split query)
 */
export async function setupSecurePrediction(request: PredictionRequest): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/prediction/setup`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to setup prediction');
  }

  return response.json();
}

/**
 * Compute partial prediction
 */
export async function computePartialPrediction(queryID: string): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/prediction/compute/${queryID}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to compute partial prediction');
  }

  return response.json();
}

/**
 * Reconstruct final prediction
 */
export async function reconstructPrediction(queryID: string): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/prediction/reconstruct/${queryID}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to reconstruct prediction');
  }

  return response.json();
}

/**
 * Get final prediction result
 */
export async function getPrediction(queryID: string): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/prediction/${queryID}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get prediction');
  }
  
  return response.json();
}

/**
 * Get aggregation configuration
 */
export async function getConfig(): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/config`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get config');
  }
  
  return response.json();
}

/**
 * Get current training round
 */
export async function getCurrentRound(): Promise<{
  currentRound: number;
  beta: number;
  collections: string[];
}> {
  const response = await fetch(`${API_BASE_URL}/api/round/current`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get current round');
  }
  
  return response.json();
}

/**
 * Get system statistics
 */
export async function getSystemStats(): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/api/stats`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get system stats');
  }
  
  return response.json();
}

/**
 * Health check
 */
export async function healthCheck(): Promise<ApiResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error('Health check failed');
  }
  
  return response.json();
}