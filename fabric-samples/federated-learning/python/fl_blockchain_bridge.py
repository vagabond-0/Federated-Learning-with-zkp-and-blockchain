"""
Bridge between Federated Learning training and Hyperledger Fabric blockchain
Integrates with your existing VPSA training code
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from fabric_client import FabricClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainFLBridge:
    """
    Bridge class that connects FL training with blockchain
    Use this in your existing FL training loop
    """
    
    def __init__(self, fabric_url: str = "http://localhost:3000"):
        self.client = FabricClient(fabric_url)
        self.round = 0
        self.registered_clients = set()
    
    def is_connected(self) -> bool:
        """Check if connected to Fabric network"""
        return self.client.health_check()
    
    def initialize(self) -> bool:
        """Initialize the blockchain ledger"""
        try:
            self.client.init_ledger()
            logger.info("Blockchain ledger initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ledger: {e}")
            return False
    
    def register_fl_client(
        self,
        client_id: str,
        domain: str,
        dataset_size: int
    ) -> bool:
        """
        Register an FL client with the blockchain
        
        Args:
            client_id: Unique identifier for this FL client
            domain: 'source' or 'target' domain
            dataset_size: Number of samples in client's local dataset
        """
        try:
            if client_id in self.registered_clients:
                logger.info(f"Client {client_id} already registered")
                return True
            
            self.client.register_client(client_id, domain, dataset_size)
            self.registered_clients.add(client_id)
            logger.info(f"Registered client {client_id} ({domain}) with {dataset_size} samples")
            return True
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False
    
    def submit_model_update(
        self,
        client_id: str,
        model: nn.Module,
        accuracy: float,
        loss: float,
        alignment_loss: float = 0.0,
        data_size: int = 0,
        prototypes: Optional[Dict[int, torch.Tensor]] = None,
        latent_features: Optional[torch.Tensor] = None
    ) -> Optional[str]:
        """
        Submit a model update to the blockchain
        
        Args:
            client_id: ID of the client submitting the model
            model: PyTorch model with trained weights
            accuracy: Model accuracy on local validation set
            loss: Training loss
            alignment_loss: VPSA alignment loss (for domain adaptation)
            data_size: Number of training samples used
            prototypes: Optional class prototypes (for VPSA)
            latent_features: Optional latent features
        
        Returns:
            Model ID if successful, None otherwise
        """
        try:
            model_id = f"model-r{self.round}-{client_id}"
            
            # Extract state dict
            state_dict = model.state_dict()
            
            # Convert prototypes
            proto_dict = None
            if prototypes is not None:
                proto_dict = {
                    str(k): v.detach().cpu().numpy() 
                    for k, v in prototypes.items()
                }
            
            # Convert latent features
            latent_dict = None
            if latent_features is not None:
                latent_dict = latent_features.detach().cpu().numpy()
            
            # Submit to blockchain
            result = self.client.submit_pytorch_model(
                model_id=model_id,
                client_id=client_id,
                round_num=self.round,
                state_dict=state_dict,
                latent_features=latent_dict,
                prototypes=proto_dict,
                accuracy=accuracy,
                loss=loss,
                alignment_loss=alignment_loss,
                data_size=data_size
            )
            
            logger.info(f"Submitted model {model_id}: {result}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to submit model update: {e}")
            return None
    
    def trigger_aggregation(self, model_ids: List[str]) -> Optional[Dict]:
        """
        Trigger model aggregation on the blockchain
        
        Args:
            model_ids: List of model IDs to aggregate
        
        Returns:
            Aggregation result or None if failed
        """
        try:
            result = self.client.aggregate_models(model_ids)
            
            if result.get('success'):
                logger.info(f"Aggregation complete: "
                           f"accuracy={result.get('globalAccuracy', 0):.4f}, "
                           f"loss={result.get('globalLoss', 0):.4f}")
                self.round += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return None
    
    def get_global_model_info(self) -> Dict:
        """Get information about the current global model"""
        return self.client.get_global_model()
    
    def get_training_history(self) -> List[Dict]:
        """Get training metrics history"""
        return self.client.get_all_metrics()
    
    def run_fl_round(
        self,
        clients: Dict[str, Tuple[nn.Module, Dict]],
        aggregate: bool = True
    ) -> Optional[Dict]:
        """
        Run a complete FL round: submit all models and optionally aggregate
        
        Args:
            clients: Dict of client_id -> (model, metrics_dict)
                     metrics_dict should contain: accuracy, loss, alignment_loss, data_size
            aggregate: Whether to trigger aggregation after submission
        
        Returns:
            Aggregation result if aggregate=True, else dict of model IDs
        """
        model_ids = []
        
        for client_id, (model, metrics) in clients.items():
            model_id = self.submit_model_update(
                client_id=client_id,
                model=model,
                accuracy=metrics.get('accuracy', 0.0),
                loss=metrics.get('loss', 0.0),
                alignment_loss=metrics.get('alignment_loss', 0.0),
                data_size=metrics.get('data_size', 0),
                prototypes=metrics.get('prototypes'),
                latent_features=metrics.get('latent_features')
            )
            
            if model_id:
                model_ids.append(model_id)
        
        if aggregate and len(model_ids) >= 2:
            return self.trigger_aggregation(model_ids)
        
        return {'model_ids': model_ids}


# Example usage with your existing VPSA code
def example_integration():
    """
    Example showing how to integrate with your existing FL training code
    """
    import torch.nn as nn
    
    # Create bridge
    bridge = BlockchainFLBridge()
    
    # Check connection
    if not bridge.is_connected():
        print("ERROR: Cannot connect to Fabric REST API")
        print("Make sure to run: cd sdk && npm run server")
        return
    
    # Initialize (only needed once)
    bridge.initialize()
    
    # Register clients
    bridge.register_fl_client("hospital-a", "source", 5000)
    bridge.register_fl_client("hospital-b", "source", 4500)
    bridge.register_fl_client("clinic-x", "target", 3000)
    bridge.register_fl_client("clinic-y", "target", 2500)
    
    # Simulate training round
    # In your actual code, replace this with your trained models
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 256)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    # Submit models (simulated)
    clients_data = {
        "hospital-a": (SimpleModel(), {
            'accuracy': 0.85,
            'loss': 0.15,
            'alignment_loss': 0.02,
            'data_size': 5000,
            'prototypes': {0: torch.randn(128), 1: torch.randn(128)}
        }),
        "hospital-b": (SimpleModel(), {
            'accuracy': 0.82,
            'loss': 0.18,
            'alignment_loss': 0.03,
            'data_size': 4500
        }),
        "clinic-x": (SimpleModel(), {
            'accuracy': 0.72,
            'loss': 0.28,
            'alignment_loss': 0.08,
            'data_size': 3000
        }),
        "clinic-y": (SimpleModel(), {
            'accuracy': 0.68,
            'loss': 0.32,
            'alignment_loss': 0.10,
            'data_size': 2500
        }),
    }
    
    # Run FL round
    result = bridge.run_fl_round(clients_data, aggregate=True)
    print(f"Round result: {result}")
    
    # Get global model info
    global_model = bridge.get_global_model_info()
    print(f"Global model: {global_model}")
    
    # Get training history
    history = bridge.get_training_history()
    print(f"Training history: {history}")


if __name__ == "__main__":
    example_integration()