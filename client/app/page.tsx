"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";

interface HealthStatus {
  status: string;
  timestamp: string;
  network: string;
  chaincode: string;
}

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:5000/health")
      .then((res) => res.json())
      .then((data) => {
        setHealth(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const features = [
    {
      title: "Client Management",
      description: "Register and manage federated learning clients",
      href: "/clients",
      icon: "👥",
    },
    {
      title: "Model Submission",
      description: "Submit local models from source and target domains",
      href: "/models",
      icon: "📤",
    },
    {
      title: "Model Aggregation",
      description: "Aggregate models using VPSA algorithm",
      href: "/aggregate",
      icon: "🔄",
    },
    {
      title: "Global Model",
      description: "View and track global model updates",
      href: "/global-model",
      icon: "🌐",
    },
    {
      title: "Training Metrics",
      description: "Monitor training progress and metrics",
      href: "/metrics",
      icon: "📊",
    },
    {
      title: "Configuration",
      description: "Manage aggregation configuration",
      href: "/config",
      icon: "⚙️",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-white mb-4">
            VPSA Federated Learning
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Virtual Prototype Semantic Alignment for Cross-Domain Federated Learning
            with Blockchain & Zero-Knowledge Proofs
          </p>
          
          {/* Health Status */}
          <div className="mt-8 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gray-800/50 border border-gray-700">
            {loading ? (
              <span className="text-gray-400">Checking network status...</span>
            ) : health?.status === "healthy" ? (
              <>
                <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-green-400">Network Connected</span>
                <span className="text-gray-500">|</span>
                <span className="text-gray-400">{health.chaincode}</span>
              </>
            ) : (
              <>
                <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                <span className="text-red-400">Network Disconnected</span>
              </>
            )}
          </div>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {features.map((feature) => (
            <Link
              key={feature.href}
              href={feature.href}
              className="group p-6 bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl hover:border-blue-500 hover:bg-gray-800/80 transition-all duration-300"
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h2 className="text-xl font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors">
                {feature.title}
              </h2>
              <p className="text-gray-400">{feature.description}</p>
            </Link>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="mt-16 text-center">
          <h3 className="text-2xl font-semibold text-white mb-6">Quick Actions</h3>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              href="/demo"
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
            >
              🚀 Run Demo Workflow
            </Link>
            <Link
              href="/clients"
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors"
            >
              ➕ Register New Client
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
