"use client";

import { useState, useEffect } from "react";

interface AggregationConfig {
  minClients?: number;
  min_clients?: number;
  sourceWeight?: number;
  source_weight?: number;
  targetWeight?: number;
  target_weight?: number;
  alignmentThreshold?: number;
  alignment_threshold?: number;
  maxRounds?: number;
  max_rounds?: number;
  learningRate?: number;
  learning_rate?: number;
}

export default function ConfigPage() {
  const [config, setConfig] = useState<AggregationConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [formData, setFormData] = useState({
    minClients: 2,
    sourceWeight: 0.6,
    targetWeight: 0.4,
    alignmentThreshold: 0.15,
    maxRounds: 10,
    learningRate: 0.01,
  });

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/config");
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
        // Update form with fetched data
        setFormData({
          minClients: data.minClients ?? data.min_clients ?? 2,
          sourceWeight: data.sourceWeight ?? data.source_weight ?? 0.6,
          targetWeight: data.targetWeight ?? data.target_weight ?? 0.4,
          alignmentThreshold: data.alignmentThreshold ?? data.alignment_threshold ?? 0.15,
          maxRounds: data.maxRounds ?? data.max_rounds ?? 10,
          learningRate: data.learningRate ?? data.learning_rate ?? 0.01,
        });
      }
    } catch (err) {
      setError("Failed to fetch configuration");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setSuccess(null);

    // Validate weights sum to 1
    if (Math.abs(formData.sourceWeight + formData.targetWeight - 1) > 0.001) {
      setError("Source weight and target weight must sum to 1.0");
      setSaving(false);
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/api/config/update", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          minClients: formData.minClients.toString(),
          sourceWeight: formData.sourceWeight.toString(),
          targetWeight: formData.targetWeight.toString(),
          alignmentThreshold: formData.alignmentThreshold.toString(),
        }),
      });

      const data = await res.json();

      if (res.ok) {
        setSuccess("Configuration updated successfully!");
        fetchConfig(); // Refresh config
      } else {
        setError(data.error || "Failed to update configuration");
      }
    } catch (err) {
      setError("Failed to update configuration");
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const resetToDefaults = () => {
    setFormData({
      minClients: 2,
      sourceWeight: 0.6,
      targetWeight: 0.4,
      alignmentThreshold: 0.15,
      maxRounds: 10,
      learningRate: 0.01,
    });
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
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white">⚙️ Configuration</h1>
          <p className="text-gray-400 mt-1">
            Manage VPSA aggregation configuration settings
          </p>
        </div>

        {/* Alerts */}
        {error && (
          <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {success && (
          <div className="bg-green-900/50 border border-green-700 rounded-lg p-4 mb-6">
            <p className="text-green-400">{success}</p>
          </div>
        )}

        {/* Current Configuration Display */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 mb-8">
          <h2 className="text-xl font-semibold text-white mb-4">
            📋 Current Configuration
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-gray-700/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm">Min Clients</p>
              <p className="text-2xl font-bold text-white">
                {config?.minClients ?? config?.min_clients ?? "N/A"}
              </p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm">Source Weight</p>
              <p className="text-2xl font-bold text-blue-400">
                {((config?.sourceWeight ?? config?.source_weight ?? 0) * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm">Target Weight</p>
              <p className="text-2xl font-bold text-purple-400">
                {((config?.targetWeight ?? config?.target_weight ?? 0) * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm">Alignment Threshold</p>
              <p className="text-2xl font-bold text-yellow-400">
                {config?.alignmentThreshold ?? config?.alignment_threshold ?? "N/A"}
              </p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm">Max Rounds</p>
              <p className="text-2xl font-bold text-green-400">
                {config?.maxRounds ?? config?.max_rounds ?? "N/A"}
              </p>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm">Learning Rate</p>
              <p className="text-2xl font-bold text-orange-400">
                {config?.learningRate ?? config?.learning_rate ?? "N/A"}
              </p>
            </div>
          </div>
        </div>

        {/* Configuration Form */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold text-white mb-6">
            ✏️ Update Configuration
          </h2>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Min Clients */}
            <div>
              <label className="block text-gray-300 mb-2">
                Minimum Clients Required
              </label>
              <input
                type="number"
                name="minClients"
                value={formData.minClients}
                onChange={handleInputChange}
                min="1"
                max="100"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              />
              <p className="text-gray-500 text-sm mt-1">
                Minimum number of clients required for aggregation
              </p>
            </div>

            {/* Weights */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-300 mb-2">
                  Source Domain Weight
                </label>
                <input
                  type="number"
                  name="sourceWeight"
                  value={formData.sourceWeight}
                  onChange={handleInputChange}
                  min="0"
                  max="1"
                  step="0.05"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
                <p className="text-gray-500 text-sm mt-1">
                  Weight for source domain models (0-1)
                </p>
              </div>
              <div>
                <label className="block text-gray-300 mb-2">
                  Target Domain Weight
                </label>
                <input
                  type="number"
                  name="targetWeight"
                  value={formData.targetWeight}
                  onChange={handleInputChange}
                  min="0"
                  max="1"
                  step="0.05"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
                <p className="text-gray-500 text-sm mt-1">
                  Weight for target domain models (0-1)
                </p>
              </div>
            </div>

            {/* Weight Balance Indicator */}
            <div className="bg-gray-700/50 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-400">Weight Balance</span>
                <span
                  className={`font-bold ${
                    Math.abs(formData.sourceWeight + formData.targetWeight - 1) < 0.001
                      ? "text-green-400"
                      : "text-red-400"
                  }`}
                >
                  {(formData.sourceWeight + formData.targetWeight).toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-l-full"
                  style={{ width: `${formData.sourceWeight * 100}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Source: {(formData.sourceWeight * 100).toFixed(0)}%</span>
                <span>Target: {(formData.targetWeight * 100).toFixed(0)}%</span>
              </div>
              {Math.abs(formData.sourceWeight + formData.targetWeight - 1) >= 0.001 && (
                <p className="text-red-400 text-sm mt-2">
                  ⚠️ Weights must sum to 1.0
                </p>
              )}
            </div>

            {/* Alignment Threshold */}
            <div>
              <label className="block text-gray-300 mb-2">
                Alignment Threshold
              </label>
              <input
                type="number"
                name="alignmentThreshold"
                value={formData.alignmentThreshold}
                onChange={handleInputChange}
                min="0"
                max="1"
                step="0.01"
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              />
              <p className="text-gray-500 text-sm mt-1">
                Minimum semantic alignment threshold for model acceptance
              </p>
            </div>

            {/* Max Rounds & Learning Rate (Read-only for now) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-300 mb-2">
                  Max Training Rounds
                </label>
                <input
                  type="number"
                  name="maxRounds"
                  value={formData.maxRounds}
                  onChange={handleInputChange}
                  min="1"
                  max="1000"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
                <p className="text-gray-500 text-sm mt-1">
                  Maximum number of training rounds
                </p>
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Learning Rate</label>
                <input
                  type="number"
                  name="learningRate"
                  value={formData.learningRate}
                  onChange={handleInputChange}
                  min="0.0001"
                  max="1"
                  step="0.001"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
                <p className="text-gray-500 text-sm mt-1">
                  Learning rate for model updates
                </p>
              </div>
            </div>

            {/* Buttons */}
            <div className="flex gap-4 pt-4">
              <button
                type="submit"
                disabled={saving}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
              >
                {saving ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></span>
                    Saving...
                  </span>
                ) : (
                  "💾 Save Configuration"
                )}
              </button>
              <button
                type="button"
                onClick={resetToDefaults}
                className="bg-gray-700 hover:bg-gray-600 text-white font-medium py-3 px-6 rounded-lg transition-colors"
              >
                🔄 Reset to Defaults
              </button>
            </div>
          </form>
        </div>

        {/* Configuration Tips */}
        <div className="mt-8 bg-blue-900/30 border border-blue-700 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-3">
            💡 Configuration Tips
          </h3>
          <ul className="text-gray-300 space-y-2 text-sm">
            <li>
              • <strong>Source Weight</strong>: Higher values prioritize knowledge from labeled source domain data.
            </li>
            <li>
              • <strong>Target Weight</strong>: Higher values prioritize adaptation to unlabeled target domain.
            </li>
            <li>
              • <strong>Alignment Threshold</strong>: Lower values allow more diverse models but may reduce quality.
            </li>
            <li>
              • <strong>Min Clients</strong>: More clients improve model generalization but increase aggregation time.
            </li>
            <li>
              • For domain adaptation, start with 60% source / 40% target weights and adjust based on results.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}