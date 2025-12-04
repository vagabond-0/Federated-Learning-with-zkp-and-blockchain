const API_BASE = "http://localhost:5000";

export interface Client {
  clientID: string;
  domain: string;
  datasetSize: number;
  registrationTime: string;
  isActive: boolean;
  roundsParticipated: number;
}

export interface LocalModel {
  modelID: string;
  clientID: string;
  weights: string;
  latentFeatures: string;
  prototypes: string;
  accuracy: number;
  loss: number;
  alignmentLoss: number;
  dataSize: number;
  round: number;
  timestamp: string;
}

export interface GlobalModel {
  modelID: string;
  aggregatedWeights: string;
  aggregatedPrototypes: string;
  globalAccuracy: number;
  globalLoss: number;
  alignmentScore: number;
  round: number;
  contributingClients: string[];
  timestamp: string;
}

export interface TrainingMetrics {
  round: number;
  globalAccuracy: number;
  globalLoss: number;
  alignmentScore: number;
  sourceAccuracy: number;
  targetAccuracy: number;
  numClients: number;
  timestamp: string;
}

export interface AggregationConfig {
  minClientsPerRound: number;
  sourceWeight: number;
  targetWeight: number;
  alignmentWeight: number;
}

// Health check
export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

// Client operations
export async function registerClient(clientID: string, domain: string, datasetSize: number) {
  const res = await fetch(`${API_BASE}/api/client/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clientID, domain, datasetSize }),
  });
  return res.json();
}

export async function getClient(clientID: string): Promise<Client> {
  const res = await fetch(`${API_BASE}/api/client/${clientID}`);
  return res.json();
}

export async function getAllClients(): Promise<Client[]> {
  const res = await fetch(`${API_BASE}/api/clients`);
  return res.json();
}

// Model operations
export async function submitLocalModel(model: {
  modelID: string;
  clientID: string;
  weights: object;
  latentFeatures: object;
  prototypes: object;
  accuracy: number;
  loss: number;
  alignmentLoss: number;
  dataSize: number;
}) {
  const res = await fetch(`${API_BASE}/api/model/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(model),
  });
  return res.json();
}

export async function getLocalModel(modelID: string): Promise<LocalModel> {
  const res = await fetch(`${API_BASE}/api/model/${modelID}`);
  return res.json();
}

export async function getModelsByRound(round: number): Promise<LocalModel[]> {
  const res = await fetch(`${API_BASE}/api/models/round/${round}`);
  return res.json();
}

// Aggregation
export async function aggregateModels(
  modelIDs: string[],
  sourceWeight = 0.6,
  targetWeight = 0.4,
  alignmentWeight = 0.1
) {
  const res = await fetch(`${API_BASE}/api/aggregate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modelIDs, sourceWeight, targetWeight, alignmentWeight }),
  });
  return res.json();
}

// Global model
export async function getGlobalModel(): Promise<GlobalModel> {
  const res = await fetch(`${API_BASE}/api/global-model`);
  return res.json();
}

export async function getGlobalModelHistory(): Promise<GlobalModel[]> {
  const res = await fetch(`${API_BASE}/api/global-model/history`);
  return res.json();
}

// Metrics
export async function getMetrics(round: number): Promise<TrainingMetrics> {
  const res = await fetch(`${API_BASE}/api/metrics/${round}`);
  return res.json();
}

export async function getAllMetrics(): Promise<TrainingMetrics[]> {
  const res = await fetch(`${API_BASE}/api/metrics`);
  return res.json();
}

// Config
export async function getConfig(): Promise<AggregationConfig> {
  const res = await fetch(`${API_BASE}/api/config`);
  return res.json();
}

export async function updateConfig(config: AggregationConfig) {
  const res = await fetch(`${API_BASE}/api/config/update`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return res.json();
}