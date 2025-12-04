"use client";

import { useState, useEffect } from "react";
import { getModelsByRound, aggregateModels, LocalModel } from "@/lib/api";

export default function AggregatePage() {
  const [round, setRound] = useState(0);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [aggregating, setAggregating] = useState(false);
  const [message, setMessage] = useState<{ type: string; text: string } | null>(null);
  const [result, setResult] = useState<any>(null);

  const [config, setConfig] = useState({
    sourceWeight: 0.6,
    targetWeight: 0.4,
    alignmentWeight: 0.1,
  });

  useEffect(() => {
    setLoading(true);
    getModelsByRound(round)
      .then((data) => setModels(Array.isArray(data) ? data : []))
      .finally(() => setLoading(false));
  }, [round]);

  const toggleModel = (modelID: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelID)
        ? prev.filter((id) => id !== modelID)
        : [...prev, modelID]
    );
  };

  const selectAll = () => {
    setSelectedModels(models.map((m) => m.modelID));
  };

  const handleAggregate = async () => {
    if (selectedModels.length < 2) {
      setMessage({ type: "error", text: "Please select at least 2 models to aggregate" });
      return;
    }

    setAggregating(true);
    setMessage(null);
    setResult(null);

    try {
      const res = await aggregateModels(
        selectedModels,
        config.sourceWeight,
        config.targetWeight,
        config.alignmentWeight
      );

      if (res.error) {
        setMessage({ type: "error", text: res.error });
      } else {
        setMessage({ type: "success", text: "Models aggregated successfully!" });
        setResult(res);
      }
    } catch (error) {
      setMessage({ type: "error", text: "Failed to aggregate models" });
    } finally {
      setAggregating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">Model Aggregation</h1>
          <p className="text-gray-400 mt-1">Aggregate local models using VPSA algorithm</p>
        </div>

        {message && (
          <div
            className={`mb-6 p-4 rounded-lg ${
              message.type === "success"
                ? "bg-green-900/50 border border-green-700 text-green-300"
                : "bg-red-900/50 border border-red-700 text-red-300"
            }`}
          >
            {message.text}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Model Selection */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-white">Select Models</h2>
                <div className="flex items-center gap-4">
                  <label className="text-gray-400 text-sm">Round:</label>
                  <input
                    type="number"
                    value={round}
                    onChange={(e) => setRound(parseInt(e.target.value) || 0)}
                    min="0"
                    className="w-20 px-3 py-1 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                  />
                  <button
                    onClick={selectAll}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-lg"
                  >
                    Select All
                  </button>
                </div>
              </div>

              {loading ? (
                <div className="text-center py-8">
                  <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
                </div>
              ) : models.length === 0 ? (
                <p className="text-gray-400 text-center py-8">No models for round {round}</p>
              ) : (
                <div className="space-y-3">
                  {models.map((model) => (
                    <div
                      key={model.modelID}
                      onClick={() => toggleModel(model.modelID)}
                      className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                        selectedModels.includes(model.modelID)
                          ? "bg-blue-900/30 border-blue-500"
                          : "bg-gray-700/50 border-gray-600 hover:border-gray-500"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-white font-medium">{model.modelID}</p>
                          <p className="text-gray-400 text-sm">{model.clientID}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-white">{(model.accuracy * 100).toFixed(1)}%</p>
                          <p className="text-gray-400 text-sm">Accuracy</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right: Config & Action */}
          <div className="space-y-6">
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Aggregation Weights</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    Source Weight: {config.sourceWeight}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.sourceWeight}
                    onChange={(e) => setConfig({ ...config, sourceWeight: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    Target Weight: {config.targetWeight}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.targetWeight}
                    onChange={(e) => setConfig({ ...config, targetWeight: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    Alignment Weight: {config.alignmentWeight}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.05"
                    value={config.alignmentWeight}
                    onChange={(e) => setConfig({ ...config, alignmentWeight: parseFloat(e.target.value) })}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Aggregate</h2>
              <p className="text-gray-400 mb-4">
                Selected: {selectedModels.length} model(s)
              </p>
              <button
                onClick={handleAggregate}
                disabled={aggregating || selectedModels.length < 2}
                className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                {aggregating ? "Aggregating..." : "🔄 Aggregate Models"}
              </button>
            </div>

            {result && (
              <div className="bg-gray-800 border border-green-700 rounded-xl p-6">
                <h2 className="text-xl font-semibold text-green-400 mb-4">✅ Result</h2>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Models Aggregated:</span>
                    <span className="text-white">{result.num_models}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Global Accuracy:</span>
                    <span className="text-white">{(result.metrics?.global_accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Global Loss:</span>
                    <span className="text-white">{result.metrics?.global_loss.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Alignment Score:</span>
                    <span className="text-white">{result.metrics?.alignment_score.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}