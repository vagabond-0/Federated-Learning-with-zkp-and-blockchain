"use client";

import { useState, useEffect } from "react";
import { getAllMetrics, TrainingMetrics } from "@/lib/api";

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAllMetrics()
      .then((data) => setMetrics(Array.isArray(data) ? data : []))
      .finally(() => setLoading(false));
  }, []);

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
          <h1 className="text-3xl font-bold text-white">Training Metrics</h1>
          <p className="text-gray-400 mt-1">Monitor training progress across rounds</p>
        </div>

        {metrics.length === 0 ? (
          <div className="text-center py-12 bg-gray-800/50 rounded-xl border border-gray-700">
            <p className="text-gray-400 text-lg">No training metrics available yet</p>
            <p className="text-gray-500 mt-2">Submit and aggregate models to see metrics</p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
                <p className="text-gray-400 text-sm">Total Rounds</p>
                <p className="text-3xl font-bold text-white">{metrics.length}</p>
              </div>
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
                <p className="text-gray-400 text-sm">Best Accuracy</p>
                <p className="text-3xl font-bold text-green-400">
                  {Math.max(...metrics.map((m) => m.globalAccuracy) || [0]) * 100}%
                </p>
              </div>
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
                <p className="text-gray-400 text-sm">Lowest Loss</p>
                <p className="text-3xl font-bold text-yellow-400">
                  {Math.min(...metrics.map((m) => m.globalLoss) || [0]).toFixed(4)}
                </p>
              </div>
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
                <p className="text-gray-400 text-sm">Avg Alignment</p>
                <p className="text-3xl font-bold text-blue-400">
                  {((metrics.reduce((a, m) => a + m.alignmentScore, 0) / metrics.length) * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Metrics Table */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Round-by-Round Metrics</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-400 border-b border-gray-700">
                      <th className="pb-3">Round</th>
                      <th className="pb-3">Global Accuracy</th>
                      <th className="pb-3">Global Loss</th>
                      <th className="pb-3">Alignment Score</th>
                      <th className="pb-3">Source Acc</th>
                      <th className="pb-3">Target Acc</th>
                      <th className="pb-3">Clients</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((m) => (
                      <tr key={m.round} className="border-b border-gray-700/50 text-white">
                        <td className="py-3 font-medium">{m.round}</td>
                        <td className="py-3">
                          <span className="text-green-400">{(m.globalAccuracy * 100).toFixed(1)}%</span>
                        </td>
                        <td className="py-3">{m.globalLoss.toFixed(4)}</td>
                        <td className="py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full"
                                style={{ width: `${m.alignmentScore * 100}%` }}
                              ></div>
                            </div>
                            <span>{(m.alignmentScore * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="py-3 text-blue-300">{(m.sourceAccuracy * 100).toFixed(1)}%</td>
                        <td className="py-3 text-purple-300">{(m.targetAccuracy * 100).toFixed(1)}%</td>
                        <td className="py-3">{m.numClients}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}