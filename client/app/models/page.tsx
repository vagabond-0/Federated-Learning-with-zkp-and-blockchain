"use client";

import { useState, useEffect } from "react";
import { getAllClients, submitLocalModel, getModelsByRound, Client, LocalModel } from "@/lib/api";

export default function ModelsPage() {
  const [clients, setClients] = useState<Client[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [round, setRound] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: string; text: string } | null>(null);

  const [formData, setFormData] = useState({
    modelID: "",
    clientID: "",
    weights: '{"layer1":[0.5,0.3],"layer2":[0.8,0.2]}',
    latentFeatures: '{"latent_dim":768,"features":[0.1,0.2]}',
    prototypes: '{"class1":[0.9,0.1],"class2":[0.2,0.8]}',
    accuracy: 0.85,
    loss: 0.15,
    alignmentLoss: 0.05,
    dataSize: 1000,
  });

  useEffect(() => {
    Promise.all([getAllClients(), getModelsByRound(round)])
      .then(([clientsData, modelsData]) => {
        setClients(Array.isArray(clientsData) ? clientsData : []);
        setModels(Array.isArray(modelsData) ? modelsData : []);
      })
      .finally(() => setLoading(false));
  }, [round]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setMessage(null);

    try {
      const result = await submitLocalModel({
        modelID: formData.modelID,
        clientID: formData.clientID,
        weights: JSON.parse(formData.weights),
        latentFeatures: JSON.parse(formData.latentFeatures),
        prototypes: JSON.parse(formData.prototypes),
        accuracy: formData.accuracy,
        loss: formData.loss,
        alignmentLoss: formData.alignmentLoss,
        dataSize: formData.dataSize,
      });

      if (result.error) {
        setMessage({ type: "error", text: result.error });
      } else {
        setMessage({ type: "success", text: `Model ${formData.modelID} submitted successfully!` });
        setShowForm(false);
        const modelsData = await getModelsByRound(round);
        setModels(Array.isArray(modelsData) ? modelsData : []);
      }
    } catch (error) {
      setMessage({ type: "error", text: "Failed to submit model" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Model Submission</h1>
            <p className="text-gray-400 mt-1">Submit local models from federated clients</p>
          </div>
          <button
            onClick={() => setShowForm(!showForm)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            {showForm ? "Cancel" : "📤 Submit Model"}
          </button>
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

        {showForm && (
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 mb-8">
            <h2 className="text-xl font-semibold text-white mb-4">Submit Local Model</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Model ID
                  </label>
                  <input
                    type="text"
                    value={formData.modelID}
                    onChange={(e) => setFormData({ ...formData, modelID: e.target.value })}
                    placeholder="e.g., model-source-r0"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Client ID
                  </label>
                  <select
                    value={formData.clientID}
                    onChange={(e) => setFormData({ ...formData, clientID: e.target.value })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    required
                  >
                    <option value="">Select a client</option>
                    {clients.map((client) => (
                      <option key={client.clientID} value={client.clientID}>
                        {client.clientID} ({client.domain})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Model Weights (JSON)
                </label>
                <textarea
                  value={formData.weights}
                  onChange={(e) => setFormData({ ...formData, weights: e.target.value })}
                  rows={3}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Latent Features (JSON)
                </label>
                <textarea
                  value={formData.latentFeatures}
                  onChange={(e) => setFormData({ ...formData, latentFeatures: e.target.value })}
                  rows={2}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Prototypes (JSON)
                </label>
                <textarea
                  value={formData.prototypes}
                  onChange={(e) => setFormData({ ...formData, prototypes: e.target.value })}
                  rows={2}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-blue-500"
                  required
                />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Accuracy</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={formData.accuracy}
                    onChange={(e) => setFormData({ ...formData, accuracy: parseFloat(e.target.value) })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Loss</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    value={formData.loss}
                    onChange={(e) => setFormData({ ...formData, loss: parseFloat(e.target.value) })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Alignment Loss</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    value={formData.alignmentLoss}
                    onChange={(e) => setFormData({ ...formData, alignmentLoss: parseFloat(e.target.value) })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Data Size</label>
                  <input
                    type="number"
                    value={formData.dataSize}
                    onChange={(e) => setFormData({ ...formData, dataSize: parseInt(e.target.value) })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                {submitting ? "Submitting..." : "Submit Model"}
              </button>
            </form>
          </div>
        )}

        <div className="mb-6 flex items-center gap-4">
          <label className="text-gray-300">View Round:</label>
          <input
            type="number"
            value={round}
            onChange={(e) => setRound(parseInt(e.target.value) || 0)}
            min="0"
            className="w-24 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          />
        </div>

        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-12 bg-gray-800/50 rounded-xl border border-gray-700">
            <p className="text-gray-400">No models submitted for round {round}</p>
          </div>
        ) : (
          <div className="space-y-4">
            {models.map((model) => (
              <div
                key={model.modelID}
                className="bg-gray-800 border border-gray-700 rounded-xl p-6"
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{model.modelID}</h3>
                    <p className="text-gray-400 text-sm">Client: {model.clientID}</p>
                  </div>
                  <span className="px-3 py-1 bg-blue-900/50 text-blue-300 rounded-full text-sm">
                    Round {model.round}
                  </span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <p className="text-gray-400">Accuracy</p>
                    <p className="text-white text-lg font-semibold">{(model.accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <p className="text-gray-400">Loss</p>
                    <p className="text-white text-lg font-semibold">{model.loss.toFixed(4)}</p>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <p className="text-gray-400">Alignment Loss</p>
                    <p className="text-white text-lg font-semibold">{model.alignmentLoss.toFixed(4)}</p>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <p className="text-gray-400">Data Size</p>
                    <p className="text-white text-lg font-semibold">{model.dataSize.toLocaleString()}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}