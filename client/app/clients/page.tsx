"use client";

import { useState, useEffect } from "react";
import { getAllClients, registerClient, Client } from "@/lib/api";

export default function ClientsPage() {
  const [clients, setClients] = useState<Client[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    clientID: "",
    domain: "source",
    datasetSize: 1000,
  });
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: string; text: string } | null>(null);

  const fetchClients = async () => {
    try {
      const data = await getAllClients();
      setClients(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error("Failed to fetch clients:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClients();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setMessage(null);

    try {
      const result = await registerClient(
        formData.clientID,
        formData.domain,
        formData.datasetSize
      );

      if (result.error) {
        setMessage({ type: "error", text: result.error });
      } else {
        setMessage({ type: "success", text: `Client ${formData.clientID} registered successfully!` });
        setFormData({ clientID: "", domain: "source", datasetSize: 1000 });
        setShowForm(false);
        fetchClients();
      }
    } catch (error) {
      setMessage({ type: "error", text: "Failed to register client" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Client Management</h1>
            <p className="text-gray-400 mt-1">Register and manage federated learning clients</p>
          </div>
          <button
            onClick={() => setShowForm(!showForm)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            {showForm ? "Cancel" : "➕ Register Client"}
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
            <h2 className="text-xl font-semibold text-white mb-4">Register New Client</h2>
            <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Client ID
                </label>
                <input
                  type="text"
                  value={formData.clientID}
                  onChange={(e) => setFormData({ ...formData, clientID: e.target.value })}
                  placeholder="e.g., client-org1-source"
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Domain
                </label>
                <select
                  value={formData.domain}
                  onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="source">Source Domain</option>
                  <option value="target">Target Domain</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Dataset Size
                </label>
                <input
                  type="number"
                  value={formData.datasetSize}
                  onChange={(e) => setFormData({ ...formData, datasetSize: parseInt(e.target.value) })}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
              <div className="md:col-span-3">
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
                >
                  {submitting ? "Registering..." : "Register Client"}
                </button>
              </div>
            </form>
          </div>
        )}

        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
            <p className="text-gray-400 mt-4">Loading clients...</p>
          </div>
        ) : clients.length === 0 ? (
          <div className="text-center py-12 bg-gray-800/50 rounded-xl border border-gray-700">
            <p className="text-gray-400 text-lg">No clients registered yet</p>
            <p className="text-gray-500 mt-2">Click "Register Client" to add your first client</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {clients.map((client) => (
              <div
                key={client.clientID}
                className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:border-gray-600 transition-colors"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white truncate">
                    {client.clientID}
                  </h3>
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${
                      client.domain === "source"
                        ? "bg-blue-900/50 text-blue-300"
                        : "bg-purple-900/50 text-purple-300"
                    }`}
                  >
                    {client.domain}
                  </span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Dataset Size:</span>
                    <span className="text-white">{client.datasetSize?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Rounds:</span>
                    <span className="text-white">{client.roundsParticipated || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className={client.isActive ? "text-green-400" : "text-gray-500"}>
                      {client.isActive ? "Active" : "Inactive"}
                    </span>
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