"use client";

import { useState } from "react";

interface StepStatus {
  status: "pending" | "running" | "success" | "error";
  message?: string;
  data?: unknown;
}

export default function DemoPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [stepStatuses, setStepStatuses] = useState<Record<number, StepStatus>>({});
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  };

  const updateStepStatus = (step: number, status: StepStatus) => {
    setStepStatuses((prev) => ({ ...prev, [step]: status }));
  };

  const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const steps = [
    {
      id: 1,
      title: "Register Source Domain Client",
      description: "Register a client from the source domain (e.g., labeled medical images)",
      endpoint: "/api/client/register",
      method: "POST",
      payload: {
        clientId: "client-source-demo",
        domainType: "source",
        datasetSize: "10000",
      },
    },
    {
      id: 2,
      title: "Register Target Domain Client",
      description: "Register a client from the target domain (e.g., unlabeled clinical images)",
      endpoint: "/api/client/register",
      method: "POST",
      payload: {
        clientId: "client-target-demo",
        domainType: "target",
        datasetSize: "8000",
      },
    },
    {
      id: 3,
      title: "Verify Registered Clients",
      description: "Query all registered clients to verify successful registration",
      endpoint: "/api/clients",
      method: "GET",
      payload: null,
    },
    {
      id: 4,
      title: "Submit Source Domain Model",
      description: "Submit a trained local model from the source domain client",
      endpoint: "/api/model/submit",
      method: "POST",
      payload: {
        modelId: "model-source-demo-r0",
        clientId: "client-source-demo",
        modelWeights: JSON.stringify({ layer1: [0.5, 0.3, 0.2], layer2: [0.8, 0.1, 0.1] }),
        semanticFeatures: JSON.stringify({ latent_dim: 768, features: [0.1, 0.2, 0.3] }),
        classPrototypes: JSON.stringify({ class1: [0.9, 0.1], class2: [0.2, 0.8] }),
        localAccuracy: "0.85",
        localLoss: "0.15",
        alignmentScore: "0.05",
        trainingRound: "0",
      },
    },
    {
      id: 5,
      title: "Submit Target Domain Model",
      description: "Submit a trained local model from the target domain client",
      endpoint: "/api/model/submit",
      method: "POST",
      payload: {
        modelId: "model-target-demo-r0",
        clientId: "client-target-demo",
        modelWeights: JSON.stringify({ layer1: [0.4, 0.4, 0.2], layer2: [0.7, 0.2, 0.1] }),
        semanticFeatures: JSON.stringify({ latent_dim: 768, features: [0.15, 0.25, 0.35] }),
        classPrototypes: JSON.stringify({ class1: [0.85, 0.15], class2: [0.25, 0.75] }),
        localAccuracy: "0.78",
        localLoss: "0.22",
        alignmentScore: "0.08",
        trainingRound: "0",
      },
    },
    {
      id: 6,
      title: "Query Submitted Models",
      description: "Retrieve local models submitted in round 0",
      endpoint: "/api/models/round/0",
      method: "GET",
      payload: null,
    },
    {
      id: 7,
      title: "Aggregate Models (VPSA)",
      description: "Perform Virtual Prototype Semantic Alignment aggregation",
      endpoint: "/api/aggregate",
      method: "POST",
      payload: {
        modelIds: ["model-source-demo-r0", "model-target-demo-r0"],
        aggregatedWeights: JSON.stringify({
          layer1: [0.45, 0.35, 0.2],
          layer2: [0.75, 0.15, 0.1],
        }),
        aggregatedPrototypes: JSON.stringify({
          class1: [0.875, 0.125],
          class2: [0.225, 0.775],
        }),
        globalAccuracy: "0.82",
        globalLoss: "0.18",
        alignmentScore: "0.92",
      },
    },
    {
      id: 8,
      title: "Query Global Model",
      description: "Retrieve the aggregated global model",
      endpoint: "/api/global-model",
      method: "GET",
      payload: null,
    },
    {
      id: 9,
      title: "Query Training Metrics",
      description: "Get training metrics for round 0",
      endpoint: "/api/metrics/0",
      method: "GET",
      payload: null,
    },
    {
      id: 10,
      title: "Update Configuration",
      description: "Modify VPSA aggregation configuration",
      endpoint: "/api/config/update",
      method: "PUT",
      payload: {
        minClients: "2",
        sourceWeight: "0.7",
        targetWeight: "0.3",
        alignmentThreshold: "0.1",
      },
    },
  ];

  const runStep = async (stepIndex: number) => {
    const step = steps[stepIndex];
    updateStepStatus(stepIndex, { status: "running" });
    addLog(`Starting: ${step.title}`);

    try {
      const options: RequestInit = {
        method: step.method,
        headers: { "Content-Type": "application/json" },
      };

      if (step.payload && step.method !== "GET") {
        options.body = JSON.stringify(step.payload);
      }

      const res = await fetch(`http://localhost:5000${step.endpoint}`, options);
      const data = await res.json();

      if (res.ok) {
        updateStepStatus(stepIndex, {
          status: "success",
          message: "Completed successfully",
          data,
        });
        addLog(`✅ Success: ${step.title}`);
      } else {
        updateStepStatus(stepIndex, {
          status: "error",
          message: data.error || "Request failed",
          data,
        });
        addLog(`❌ Error: ${step.title} - ${data.error}`);
      }
    } catch (err) {
      updateStepStatus(stepIndex, {
        status: "error",
        message: "Network error",
      });
      addLog(`❌ Error: ${step.title} - Network error`);
    }
  };

  const runAllSteps = async () => {
    setIsRunning(true);
    setStepStatuses({});
    setLogs([]);
    addLog("🚀 Starting VPSA Federated Learning Demo...");

    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      await runStep(i);
      await delay(1500); // Wait between steps for visibility
    }

    addLog("🎉 Demo completed!");
    setIsRunning(false);
  };

  const runSingleStep = async (stepIndex: number) => {
    setIsRunning(true);
    setCurrentStep(stepIndex);
    await runStep(stepIndex);
    setIsRunning(false);
  };

  const resetDemo = () => {
    setCurrentStep(0);
    setStepStatuses({});
    setLogs([]);
  };

  const getStepIcon = (index: number) => {
    const status = stepStatuses[index];
    if (!status) return "⚪";
    switch (status.status) {
      case "running":
        return "🔄";
      case "success":
        return "✅";
      case "error":
        return "❌";
      default:
        return "⚪";
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">🎮 Interactive Demo</h1>
          <p className="text-gray-400 mt-1">
            Step-by-step guide to VPSA Federated Learning with Blockchain
          </p>
        </div>

        {/* Control Panel */}
        <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 border border-blue-700 rounded-xl p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white">Demo Controls</h2>
              <p className="text-gray-400 text-sm">
                Run the complete workflow or execute individual steps
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={runAllSteps}
                disabled={isRunning}
                className="bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:cursor-not-allowed text-white font-medium py-2 px-6 rounded-lg transition-colors flex items-center gap-2"
              >
                {isRunning ? (
                  <>
                    <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></span>
                    Running...
                  </>
                ) : (
                  <>▶️ Run All Steps</>
                )}
              </button>
              <button
                onClick={resetDemo}
                disabled={isRunning}
                className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                🔄 Reset
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Steps Panel */}
          <div className="lg:col-span-2 space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">
              📋 Workflow Steps
            </h3>
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={`bg-gray-800 border rounded-xl p-4 transition-all ${
                  currentStep === index && isRunning
                    ? "border-blue-500 shadow-lg shadow-blue-500/20"
                    : stepStatuses[index]?.status === "success"
                    ? "border-green-700"
                    : stepStatuses[index]?.status === "error"
                    ? "border-red-700"
                    : "border-gray-700"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className="text-2xl">{getStepIcon(index)}</div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="text-white font-medium">
                        Step {step.id}: {step.title}
                      </h4>
                      <button
                        onClick={() => runSingleStep(index)}
                        disabled={isRunning}
                        className="text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-3 py-1 rounded transition-colors"
                      >
                        Run
                      </button>
                    </div>
                    <p className="text-gray-400 text-sm mt-1">
                      {step.description}
                    </p>
                    <div className="mt-2 flex items-center gap-2 text-xs">
                      <span
                        className={`px-2 py-0.5 rounded ${
                          step.method === "GET"
                            ? "bg-green-900 text-green-400"
                            : step.method === "POST"
                            ? "bg-blue-900 text-blue-400"
                            : "bg-yellow-900 text-yellow-400"
                        }`}
                      >
                        {step.method}
                      </span>
                      <span className="text-gray-500">{step.endpoint}</span>
                    </div>

                    {/* Payload Preview */}
                    {step.payload && (
                      <details className="mt-2">
                        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                          View Payload
                        </summary>
                        <pre className="mt-2 text-xs bg-gray-900 rounded p-2 overflow-x-auto text-gray-400">
                          {JSON.stringify(step.payload, null, 2)}
                        </pre>
                      </details>
                    )}

                    {/* Response Preview */}
                    {stepStatuses[index]?.data && (
                      <details className="mt-2" open>
                        <summary className="text-xs text-green-500 cursor-pointer hover:text-green-400">
                          View Response
                        </summary>
                        <pre className="mt-2 text-xs bg-gray-900 rounded p-2 overflow-x-auto text-green-400">
                          {JSON.stringify(stepStatuses[index].data, null, 2)}
                        </pre>
                      </details>
                    )}

                    {/* Error Message */}
                    {stepStatuses[index]?.status === "error" && (
                      <p className="mt-2 text-sm text-red-400">
                        ⚠️ {stepStatuses[index].message}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Progress */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <h3 className="text-lg font-semibold text-white mb-4">
                📊 Progress
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Completed</span>
                    <span className="text-white">
                      {Object.values(stepStatuses).filter((s) => s.status === "success").length} / {steps.length}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all"
                      style={{
                        width: `${
                          (Object.values(stepStatuses).filter((s) => s.status === "success").length / steps.length) * 100
                        }%`,
                      }}
                    ></div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-green-900/30 rounded p-2">
                    <p className="text-2xl font-bold text-green-400">
                      {Object.values(stepStatuses).filter((s) => s.status === "success").length}
                    </p>
                    <p className="text-xs text-gray-400">Success</p>
                  </div>
                  <div className="bg-red-900/30 rounded p-2">
                    <p className="text-2xl font-bold text-red-400">
                      {Object.values(stepStatuses).filter((s) => s.status === "error").length}
                    </p>
                    <p className="text-xs text-gray-400">Errors</p>
                  </div>
                  <div className="bg-gray-700/30 rounded p-2">
                    <p className="text-2xl font-bold text-gray-400">
                      {steps.length - Object.keys(stepStatuses).length}
                    </p>
                    <p className="text-xs text-gray-400">Pending</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Logs */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <h3 className="text-lg font-semibold text-white mb-4">
                📜 Activity Log
              </h3>
              <div className="bg-gray-900 rounded-lg p-3 h-64 overflow-y-auto font-mono text-xs">
                {logs.length === 0 ? (
                  <p className="text-gray-500">No activity yet. Run the demo to see logs.</p>
                ) : (
                  logs.map((log, index) => (
                    <p
                      key={index}
                      className={`mb-1 ${
                        log.includes("✅")
                          ? "text-green-400"
                          : log.includes("❌")
                          ? "text-red-400"
                          : log.includes("🚀") || log.includes("🎉")
                          ? "text-blue-400"
                          : "text-gray-400"
                      }`}
                    >
                      {log}
                    </p>
                  ))
                )}
              </div>
            </div>

            {/* Quick Links */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <h3 className="text-lg font-semibold text-white mb-4">
                🔗 Quick Links
              </h3>
              <div className="space-y-2">
                <a
                  href="/clients"
                  className="block bg-gray-700 hover:bg-gray-600 rounded-lg p-3 text-white transition-colors"
                >
                  👥 View Clients
                </a>
                <a
                  href="/models"
                  className="block bg-gray-700 hover:bg-gray-600 rounded-lg p-3 text-white transition-colors"
                >
                  📤 Model Submission
                </a>
                <a
                  href="/global-model"
                  className="block bg-gray-700 hover:bg-gray-600 rounded-lg p-3 text-white transition-colors"
                >
                  🌐 Global Model
                </a>
                <a
                  href="/metrics"
                  className="block bg-gray-700 hover:bg-gray-600 rounded-lg p-3 text-white transition-colors"
                >
                  📊 Metrics
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* User Guide Section */}
        <div className="mt-12">
          <h2 className="text-2xl font-bold text-white mb-6">
            📖 User Guide
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* What is VPSA */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">
                🎯 What is VPSA?
              </h3>
              <p className="text-gray-300 mb-4">
                <strong>Virtual Prototype Semantic Alignment (VPSA)</strong> is an advanced
                federated learning technique designed for cross-domain learning scenarios
                where data distributions differ between clients.
              </p>
              <ul className="text-gray-400 space-y-2 text-sm">
                <li>• <strong>Source Domain:</strong> Clients with labeled data</li>
                <li>• <strong>Target Domain:</strong> Clients with unlabeled or differently distributed data</li>
                <li>• <strong>Semantic Alignment:</strong> Bridges the gap between domains using prototype matching</li>
                <li>• <strong>Privacy-Preserving:</strong> Only model updates shared, not raw data</li>
              </ul>
            </div>

            {/* Why Blockchain */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">
                ⛓️ Why Blockchain?
              </h3>
              <p className="text-gray-300 mb-4">
                Hyperledger Fabric provides a secure, immutable ledger for federated learning:
              </p>
              <ul className="text-gray-400 space-y-2 text-sm">
                <li>• <strong>Immutable Audit Trail:</strong> All model updates are permanently recorded</li>
                <li>• <strong>Transparency:</strong> Every client can verify aggregation fairness</li>
                <li>• <strong>Trust:</strong> No single point of failure or manipulation</li>
                <li>• <strong>Smart Contracts:</strong> Automated, verifiable aggregation logic</li>
                <li>• <strong>Access Control:</strong> Permissioned network ensures authorized participation</li>
              </ul>
            </div>

            {/* Workflow Overview */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">
                🔄 Workflow Overview
              </h3>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">1</div>
                  <div>
                    <p className="text-white font-medium">Client Registration</p>
                    <p className="text-gray-400 text-sm">Clients register with their domain type and dataset info</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">2</div>
                  <div>
                    <p className="text-white font-medium">Local Training</p>
                    <p className="text-gray-400 text-sm">Each client trains on their local data</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">3</div>
                  <div>
                    <p className="text-white font-medium">Model Submission</p>
                    <p className="text-gray-400 text-sm">Clients submit model weights and semantic features</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">4</div>
                  <div>
                    <p className="text-white font-medium">VPSA Aggregation</p>
                    <p className="text-gray-400 text-sm">Server aggregates models with semantic alignment</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">5</div>
                  <div>
                    <p className="text-white font-medium">Global Model Update</p>
                    <p className="text-gray-400 text-sm">Updated global model is distributed to all clients</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Concepts */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">
                💡 Key Concepts
              </h3>
              <div className="space-y-4">
                <div>
                  <p className="text-white font-medium">Model Weights</p>
                  <p className="text-gray-400 text-sm">
                    Neural network parameters learned during local training
                  </p>
                </div>
                <div>
                  <p className="text-white font-medium">Semantic Features</p>
                  <p className="text-gray-400 text-sm">
                    High-level representations that capture domain-invariant knowledge
                  </p>
                </div>
                <div>
                  <p className="text-white font-medium">Class Prototypes</p>
                  <p className="text-gray-400 text-sm">
                    Representative embeddings for each class used in alignment
                  </p>
                </div>
                <div>
                  <p className="text-white font-medium">Alignment Score</p>
                  <p className="text-gray-400 text-sm">
                    Measures how well source and target domains are aligned (0-1)
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* API Reference */}
          <div className="mt-8 bg-gray-800 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-semibold text-white mb-4">
              📚 API Reference
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-700">
                    <th className="pb-3">Endpoint</th>
                    <th className="pb-3">Method</th>
                    <th className="pb-3">Description</th>
                  </tr>
                </thead>
                <tbody className="text-gray-300">
                  <tr className="border-b border-gray-700/50">
                    <td className="py-3 font-mono text-blue-400">/api/client/register</td>
                    <td><span className="bg-blue-900 text-blue-400 px-2 py-0.5 rounded text-xs">POST</span></td>
                    <td>Register a new federated learning client</td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-3 font-mono text-blue-400">/api/clients</td>
                    <td><span className="bg-green-900 text-green-400 px-2 py-0.5 rounded text-xs">GET</span></td>
                    <td>Get all registered clients</td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-3 font-mono text-blue-400">/api/model/submit</td>
                    <td><span className="bg-blue-900 text-blue-400 px-2 py-0.5 rounded text-xs">POST</span></td>
                    <td>Submit a local model update</td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-3 font-mono text-blue-400">/api/aggregate</td>
                    <td><span className="bg-blue-900 text-blue-400 px-2 py-0.5 rounded text-xs">POST</span></td>
                    <td>Perform VPSA model aggregation</td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-3 font-mono text-blue-400">/api/global-model</td>
                    <td><span className="bg-green-900 text-green-400 px-2 py-0.5 rounded text-xs">GET</span></td>
                    <td>Get the current global model</td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-3 font-mono text-blue-400">/api/metrics/{"{round}"}</td>
                    <td><span className="bg-green-900 text-green-400 px-2 py-0.5 rounded text-xs">GET</span></td>
                    <td>Get metrics for a specific round</td>
                  </tr>
                  <tr>
                    <td className="py-3 font-mono text-blue-400">/api/config/update</td>
                    <td><span className="bg-yellow-900 text-yellow-400 px-2 py-0.5 rounded text-xs">PUT</span></td>
                    <td>Update aggregation configuration</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}