"""
Test script for Python-Fabric integration
"""

from fabric_client import FabricClient
import numpy as np

def main():
    print("=" * 50)
    print("PYTHON-FABRIC INTEGRATION TEST")
    print("=" * 50)
    
    client = FabricClient("http://localhost:3000")
    
    # Test 1: Health check
    print("\n--- Test 1: Health Check ---")
    if client.health_check():
        print("✓ REST server is connected to Fabric")
    else:
        print("✗ Cannot connect to REST server")
        print("  Run: cd ../sdk && npm run server")
        return
    
    # Test 2: Get global model
    print("\n--- Test 2: Get Global Model ---")
    try:
        model = client.get_global_model()
        print(f"✓ Global model: version={model.get('version')}, "
              f"accuracy={model.get('accuracy', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: Get config
    print("\n--- Test 3: Get Config ---")
    try:
        config = client.get_config()
        print(f"✓ Config: minClients={config.get('minClients')}, "
              f"sourceWeight={config.get('sourceWeight')}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 4: Register clients
    print("\n--- Test 4: Register Clients ---")
    try:
        client.register_client("py-client-source", "source", 1000)
        client.register_client("py-client-target", "target", 800)
        print("✓ Clients registered")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test 5: Get all clients
    print("\n--- Test 5: Get All Clients ---")
    try:
        clients = client.get_all_clients()
        print(f"✓ Found {len(clients)} clients:")
        for c in clients:
            print(f"    - {c.get('clientID')} ({c.get('domain')})")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: Submit models
    print("\n--- Test 6: Submit Local Models ---")
    try:
        # Create mock weights
        weights = {
            'layer1': np.random.randn(100).tolist(),
            'layer2': np.random.randn(50).tolist()
        }
        latent = {'z': np.random.randn(32).tolist()}
        proto = {'class0': np.random.randn(16).tolist()}
        
        result1 = client.submit_local_model(
            "py-model-1", "py-client-source", 0,
            weights, latent, proto,
            0.78, 0.22, 0.05, 1000
        )
        print(f"✓ Submitted py-model-1: {result1.get('message')}")
        
        result2 = client.submit_local_model(
            "py-model-2", "py-client-target", 0,
            weights, latent, proto,
            0.65, 0.35, 0.12, 800
        )
        print(f"✓ Submitted py-model-2: {result2.get('message')}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 7: Aggregate
    print("\n--- Test 7: Aggregate Models ---")
    try:
        result = client.aggregate_models(["py-model-1", "py-model-2"])
        print(f"✓ Aggregation result:")
        print(f"    Success: {result.get('success')}")
        print(f"    Accuracy: {result.get('globalAccuracy', 0):.4f}")
        print(f"    Loss: {result.get('globalLoss', 0):.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 8: Get metrics
    print("\n--- Test 8: Get Metrics ---")
    try:
        metrics = client.get_all_metrics()
        print(f"✓ Found {len(metrics)} training rounds")
        for m in metrics:
            print(f"    Round {m.get('round')}: "
                  f"accuracy={m.get('globalAccuracy', 0):.4f}")
    except Exception as e:
        print(f"  Note: {e}")
    
    print("\n" + "=" * 50)
    print("TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()