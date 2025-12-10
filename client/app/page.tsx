"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  getSystemStats,
  getGlobalModel,
  getCurrentRound,
  healthCheck,
} from "@/lib/api";

export default function Home() {
  const [stats, setStats] = useState<any>(null);
  const [globalModel, setGlobalModel] = useState<any>(null);
  const [currentRound, setCurrentRound] = useState<any>(null);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);

      // Health check
      try {
        await healthCheck();
        setIsHealthy(true);
      } catch {
        setIsHealthy(false);
      }

      // Load stats
      try {
        const statsData = await getSystemStats();
        setStats(statsData);
      } catch (error) {
        console.error("Failed to load stats:", error);
      }

      // Load global model
      try {
        const modelData = await getGlobalModel();
        setGlobalModel(modelData);
      } catch (error) {
        console.error("Failed to load global model:", error);
      }

      // Load current round
      try {
        const roundData = await getCurrentRound();
        setCurrentRound(roundData);
      } catch (error) {
        console.error("Failed to load current round:", error);
      }
    } catch (error) {
      console.error("Error loading dashboard:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Federated Learning with ZKP & Blockchain
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Privacy-preserving collaborative machine learning using Hyperledger
            Fabric, VPSA aggregation, and secure multi-party computation
          </p>
        </div>

        {/* Health Status */}
        <div className="mb-8">
          <div
            className={`p-4 rounded-lg ${
              isHealthy === null
                ? "bg-gray-100"
                : isHealthy
                ? "bg-green-100 border border-green-200"
                : "bg-red-100 border border-red-200"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">
                  {isHealthy === null ? "⏳" : isHealthy ? "✅" : "❌"}
                </span>
                <div>
                  <p
                    className={`font-semibold ${
                      isHealthy === null
                        ? "text-gray-700"
                        : isHealthy
                        ? "text-green-900"
                        : "text-red-900"
                    }`}
                  >
                    {isHealthy === null
                      ? "Checking connection..."
                      : isHealthy
                      ? "Backend Connected"
                      : "Backend Offline"}
                  </p>
                  <p
                    className={`text-sm ${
                      isHealthy === null
                        ? "text-gray-600"
                        : isHealthy
                        ? "text-green-700"
                        : "text-red-700"
                    }`}
                  >
                    {isHealthy === null
                      ? "Please wait..."
                      : isHealthy
                      ? "Hyperledger Fabric network is running"
                      : "Please start the Flask backend"}
                  </p>
                </div>
              </div>
              <button
                onClick={loadDashboardData}
                className="px-4 py-2 bg-white rounded-md shadow text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Main Actions */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Train Model Card */}
          <Link href="/train">
            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow cursor-pointer border-2 border-transparent hover:border-blue-500">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-2xl">
                  🧠
                </div>
                <h2 className="text-2xl font-bold text-gray-900 ml-4">
                  Train Model
                </h2>
              </div>
              <p className="text-gray-600 mb-4">
                Train a local model using your data, extract weights, and submit
                to the federated learning network via blockchain.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-blue-50 text-blue-700 text-xs rounded-full font-medium">
                  In-browser Training
                </span>
                <span className="px-3 py-1 bg-green-50 text-green-700 text-xs rounded-full font-medium">
                  TensorFlow.js
                </span>
                <span className="px-3 py-1 bg-purple-50 text-purple-700 text-xs rounded-full font-medium">
                  Private Collections
                </span>
              </div>
            </div>
          </Link>

          {/* Secure Prediction Card */}
          <Link href="/predict">
            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow cursor-pointer border-2 border-transparent hover:border-green-500">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-2xl">
                  🔮
                </div>
                <h2 className="text-2xl font-bold text-gray-900 ml-4">
                  Secure Prediction
                </h2>
              </div>
              <p className="text-gray-600 mb-4">
                Make privacy-preserving predictions using the global model with
                secret sharing and secure multi-party computation.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-green-50 text-green-700 text-xs rounded-full font-medium">
                  Secret Sharing
                </span>
                <span className="px-3 py-1 bg-yellow-50 text-yellow-700 text-xs rounded-full font-medium">
                  MPC
                </span>
                <span className="px-3 py-1 bg-red-50 text-red-700 text-xs rounded-full font-medium">
                  Zero Knowledge
                </span>
              </div>
            </div>
          </Link>
        </div>

        {/* System Statistics */}
        {!loading && stats && (
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {/* Global Model Info */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Global Model
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Version:</span>
                  <span className="font-semibold text-gray-900">
                    {stats.globalModel?.version || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Round:</span>
                  <span className="font-semibold text-gray-900">
                    {stats.globalModel?.round || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Weights:</span>
                  <span
                    className={`font-semibold ${
                      stats.globalModel?.hasWeights
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    {stats.globalModel?.hasWeights
                      ? "✓ Available"
                      : "✗ Not Available"}
                  </span>
                </div>
              </div>
            </div>

            {/* Configuration */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Configuration
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Current Round:</span>
                  <span className="font-semibold text-gray-900">
                    {stats.config?.currentRound || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Beta (Trimming):</span>
                  <span className="font-semibold text-gray-900">
                    {stats.config?.beta || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Collections:</span>
                  <span className="font-semibold text-gray-900">
                    {stats.config?.collections?.length || 0}
                  </span>
                </div>
              </div>
            </div>

            {/* Features */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Features
              </h3>
              <div className="space-y-2">
                {stats.features?.privateDataCollections && (
                  <div className="flex items-center text-sm">
                    <span className="text-green-500 mr-2">✓</span>
                    <span className="text-gray-700">Private Data Collections</span>
                  </div>
                )}
                {stats.features?.vpsaAggregation && (
                  <div className="flex items-center text-sm">
                    <span className="text-green-500 mr-2">✓</span>
                    <span className="text-gray-700">VPSA Aggregation</span>
                  </div>
                )}
                {stats.features?.securePrediction && (
                  <div className="flex items-center text-sm">
                    <span className="text-green-500 mr-2">✓</span>
                    <span className="text-gray-700">Secure Prediction</span>
                  </div>
                )}
                {stats.features?.coordinateTrimming && (
                  <div className="flex items-center text-sm">
                    <span className="text-green-500 mr-2">✓</span>
                    <span className="text-gray-700">Coordinate Trimming</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            <p className="mt-4 text-gray-600">Loading system information...</p>
          </div>
        )}

        {/* Information Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {/* Privacy Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-3xl mb-3">🔒</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Privacy-Preserving
            </h3>
            <p className="text-sm text-gray-600">
              Model weights are split into private collections. No single
              organization sees complete data.
            </p>
          </div>

          {/* Blockchain Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-3xl mb-3">⛓️</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Blockchain-Based
            </h3>
            <p className="text-sm text-gray-600">
              All operations are recorded on Hyperledger Fabric for transparency
              and immutability.
            </p>
          </div>

          {/* Federated Learning Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-3xl mb-3">🤝</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Federated Learning
            </h3>
            <p className="text-sm text-gray-600">
              Collaborative training without sharing raw data. VPSA ensures robust
              aggregation.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>Built with Next.js, TensorFlow.js, Hyperledger Fabric, and Flask</p>
          <p className="mt-2">
            Powered by VPSA (Verifiable Privacy-preserving Secure Aggregation)
          </p>
        </div>
      </div>
    </div>
  );
}
