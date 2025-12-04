"use client";

import { useState, useEffect } from "react";

interface FabricTimestamp {
  seconds?: number;
  nanos?: number;
}

interface GlobalModel {
  modelId?: string;
  round?: number;
  globalAccuracy?: number;
  global_accuracy?: number;
  globalLoss?: number;
  global_loss?: number;
  alignmentScore?: number;
  alignment_score?: number;
  contributingClients?: string[];
  contributing_clients?: string[];
  aggregatedWeights?: string;
  aggregated_weights?: string;
  aggregatedPrototypes?: string;
  aggregated_prototypes?: string;
  timestamp?: string | FabricTimestamp;
  updatedAt?: string | FabricTimestamp;
}

export default function GlobalModelPage() {
  const [globalModel, setGlobalModel] = useState<GlobalModel | null>(null);
  const [history, setHistory] = useState<GlobalModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelRes, historyRes] = await Promise.all([
          fetch("http://localhost:5000/api/global-model"),
          fetch("http://localhost:5000/api/global-model/history"),
        ]);

        if (modelRes.ok) {
          const modelData = await modelRes.json();
          setGlobalModel(modelData);
        }

        if (historyRes.ok) {
          const historyData = await historyRes.json();
          setHistory(Array.isArray(historyData) ? historyData : []);
        }
      } catch (err) {
        setError("Failed to fetch global model data");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Helper function to format timestamp
  const formatTimestamp = (ts: string | FabricTimestamp | undefined): string => {
    if (!ts) return "N/A";
    
    // If it's already a string, return it
    if (typeof ts === "string") {
      return ts;
    }
    
    // If it's a Fabric timestamp object {seconds, nanos}
    if (typeof ts === "object" && "seconds" in ts) {
      const date = new Date((ts.seconds || 0) * 1000);
      return date.toLocaleString();
    }
    
    return "N/A";
  };

  // Helper functions to handle both camelCase and snake_case field names
  const getRound = (model: GlobalModel) => model.round ?? 0;
  const getAccuracy = (model: GlobalModel) => {
    const acc = model.globalAccuracy ?? model.global_accuracy;
    return typeof acc === "number" ? acc : 0;
  };
  const getLoss = (model: GlobalModel) => {
    const loss = model.globalLoss ?? model.global_loss;
    return typeof loss === "number" ? loss : 0;
  };
  const getAlignmentScore = (model: GlobalModel) => {
    const score = model.alignmentScore ?? model.alignment_score;
    return typeof score === "number" ? score : 0;
  };
  const getContributingClients = (model: GlobalModel): string[] => {
    const clients = model.contributingClients ?? model.contributing_clients;
    return Array.isArray(clients) ? clients : [];
  };
  const getTimestamp = (model: GlobalModel): string => {
    return formatTimestamp(model.timestamp ?? model.updatedAt);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Global Model</h1>
          <p className="text-gray-400 mt-1">
            View the current global model and history
          </p>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Current Global Model */}
        {globalModel ? (
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 border border-blue-700 rounded-xl p-6 mb-8">
            <h2 className="text-xl font-semibold text-white mb-4">
              🌐 Current Global Model
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Round</p>
                <p className="text-3xl font-bold text-white">
                  {getRound(globalModel)}
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Global Accuracy</p>
                <p className="text-3xl font-bold text-green-400">
                  {(getAccuracy(globalModel) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Global Loss</p>
                <p className="text-3xl font-bold text-yellow-400">
                  {getLoss(globalModel).toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Alignment Score</p>
                <p className="text-3xl font-bold text-blue-400">
                  {(getAlignmentScore(globalModel) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            <div className="mt-4 text-sm text-gray-400">
              <p>
                Contributing Clients:{" "}
                {getContributingClients(globalModel).join(", ") || "N/A"}
              </p>
              <p>Last Updated: {getTimestamp(globalModel)}</p>
            </div>
          </div>
        ) : (
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 mb-8">
            <p className="text-gray-400 text-center">
              No global model available yet. Run aggregation to create one.
            </p>
          </div>
        )}

        {/* Model History */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold text-white mb-4">
            📜 Model History
          </h2>
          {history.length === 0 ? (
            <p className="text-gray-400">No history available</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-700">
                    <th className="pb-3">Round</th>
                    <th className="pb-3">Accuracy</th>
                    <th className="pb-3">Loss</th>
                    <th className="pb-3">Alignment</th>
                    <th className="pb-3">Clients</th>
                    <th className="pb-3">Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((model, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-gray-700/50 text-white"
                    >
                      <td className="py-3">{getRound(model)}</td>
                      <td className="py-3 text-green-400">
                        {(getAccuracy(model) * 100).toFixed(1)}%
                      </td>
                      <td className="py-3">{getLoss(model).toFixed(4)}</td>
                      <td className="py-3">
                        {(getAlignmentScore(model) * 100).toFixed(1)}%
                      </td>
                      <td className="py-3">
                        {getContributingClients(model).length}
                      </td>
                      <td className="py-3 text-gray-400">
                        {getTimestamp(model)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}