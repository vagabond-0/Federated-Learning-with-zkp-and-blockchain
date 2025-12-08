"""
Fabric REST Client for VPSA Federated Learning
Communicates with the Node.js REST API server to interact with blockchain
"""

import requests
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FabricClient:
    """
    Python client for interacting with VPSA Fabric chaincode via REST API
    """
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to REST API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            elif method == 'PUT':
                response = self.session.put(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if the REST server is running and connected to Fabric"""
        try:
            result = self._request('GET', '/health')
            return result.get('status') == 'ok'
        except:
            return False
    
    def init_ledger(self) -> Dict:
        """Initialize the ledger"""
        return self._request('POST', '/api/init')
    
    def register_client(self, client_id: str, domain: str, dataset_size: int) -> Dict:
        """
        Register a new FL client
        
        Args:
            client_id: Unique client identifier
            domain: 'source' or 'target'
            dataset_size: Number of samples in client's dataset
        """
        data = {
            'clientID': client_id,
            'domain': domain,
            'datasetSize': dataset_size
        }
        return self._request('POST', '/api/clients', data)
    
    def get_all_clients(self) -> List[Dict]:
        """Get all registered clients"""
        result = self._request('GET', '/api/clients')
        return result.get('clients', [])
    
    def submit_local_model(
        self,
        model_id: str,
        client_id: str,
        round_num: int,
        weights: Dict[str, List[float]],
        latent_features: Dict[str, List[float]],
        prototypes: Dict[str, List[float]],
        accuracy: float,
        loss: float,
        alignment_loss: float,
        data_size: int
    ) -> Dict:
        """
        Submit a local model to the blockchain
        
        Args:
            model_id: Unique model identifier
            client_id: Client that trained this model
            round_num: Current training round
            weights: Model weights as dict of layer_name -> weight_list
            latent_features: Latent feature vectors
            prototypes: Prototype vectors for each class
            accuracy: Model accuracy on local data
            loss: Model loss
            alignment_loss: VPSA alignment loss
            data_size: Size of training data
        """
        # Create payload
        payload = {
            'weights': weights,
            'latentFeatures': latent_features,
            'prototypes': prototypes,
            'accuracy': accuracy,
            'loss': loss,
            'alignmentLoss': alignment_loss,
            'dataSize': data_size
        }
        
        # Compute hash for verification
        payload_json = json.dumps(payload, sort_keys=True)
        data_hash = hashlib.sha256(payload_json.encode()).hexdigest()[:16]
        
        data = {
            'modelID': model_id,
            'clientID': client_id,
            'round': round_num,
            'dataHash': data_hash,
            'payload': payload
        }
        
        logger.info(f"Submitting model {model_id} from {client_id}")
        return self._request('POST', '/api/models/submit', data)
    
    def submit_pytorch_model(
        self,
        model_id: str,
        client_id: str,
        round_num: int,
        state_dict: Dict,
        latent_features: Optional[np.ndarray] = None,
        prototypes: Optional[Dict[str, np.ndarray]] = None,
        accuracy: float = 0.0,
        loss: float = 0.0,
        alignment_loss: float = 0.0,
        data_size: int = 0
    ) -> Dict:
        """
        Submit a PyTorch model to the blockchain
        
        Args:
            model_id: Unique model identifier
            client_id: Client that trained this model
            round_num: Current training round
            state_dict: PyTorch model state dict
            latent_features: Optional latent features as numpy array
            prototypes: Optional dict of class_name -> prototype numpy array
            accuracy: Model accuracy
            loss: Model loss
            alignment_loss: VPSA alignment loss
            data_size: Size of training data
        """
        # Convert state dict to serializable format
        weights = {}
        for key, value in state_dict.items():
            if hasattr(value, 'cpu'):
                # PyTorch tensor
                arr = value.cpu().numpy().flatten().tolist()
            elif isinstance(value, np.ndarray):
                arr = value.flatten().tolist()
            else:
                arr = list(value)
            
            # Limit precision to reduce size
            weights[key] = [round(float(x), 6) for x in arr]
        
        # Convert latent features
        latent_dict = {}
        if latent_features is not None:
            if isinstance(latent_features, np.ndarray):
                latent_dict['z'] = latent_features.flatten().tolist()
            elif isinstance(latent_features, dict):
                for k, v in latent_features.items():
                    if isinstance(v, np.ndarray):
                        latent_dict[k] = v.flatten().tolist()
                    else:
                        latent_dict[k] = list(v)
        
        # Convert prototypes
        proto_dict = {}
        if prototypes is not None:
            for cls, proto in prototypes.items():
                if isinstance(proto, np.ndarray):
                    proto_dict[str(cls)] = proto.flatten().tolist()
                else:
                    proto_dict[str(cls)] = list(proto)
        
        return self.submit_local_model(
            model_id, client_id, round_num,
            weights, latent_dict, proto_dict,
            accuracy, loss, alignment_loss, data_size
        )
    
    def get_local_model(self, model_id: str) -> Dict:
        """Get local model metadata"""
        result = self._request('GET', f'/api/models/{model_id}')
        return result.get('model', {})
    
    def aggregate_models(self, model_ids: List[str]) -> Dict:
        """
        Trigger model aggregation on the blockchain
        
        Args:
            model_ids: List of model IDs to aggregate
        
        Returns:
            Aggregation result with global accuracy, loss, etc.
        """
        data = {'modelIDs': model_ids}
        result = self._request('POST', '/api/aggregate', data)
        return result.get('result', {})
    
    def get_global_model(self) -> Dict:
        """Get the current global model metadata"""
        result = self._request('GET', '/api/global-model')
        return result.get('model', {})
    
    def get_config(self) -> Dict:
        """Get aggregation configuration"""
        result = self._request('GET', '/api/config')
        return result.get('config', {})
    
    def update_config(
        self,
        min_clients: int,
        source_weight: float,
        target_weight: float,
        alignment_weight: float
    ) -> Dict:
        """Update aggregation configuration"""
        data = {
            'minClients': min_clients,
            'sourceWeight': source_weight,
            'targetWeight': target_weight,
            'alignmentWeight': alignment_weight
        }
        return self._request('PUT', '/api/config', data)
    
    def get_training_metrics(self, round_num: int) -> Dict:
        """Get training metrics for a specific round"""
        result = self._request('GET', f'/api/metrics/{round_num}')
        return result.get('metrics', {})
    
    def get_all_metrics(self) -> List[Dict]:
        """Get all training metrics"""
        result = self._request('GET', '/api/metrics')
        return result.get('metrics', [])