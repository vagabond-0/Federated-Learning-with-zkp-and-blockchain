package main

import (
    "encoding/json"
    "fmt"
    "time"

    "github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// VPSAContract provides functions for VPSA-based federated learning
type VPSAContract struct {
    contractapi.Contract
}

// Client represents a participating client/peer in federated learning
type Client struct {
    ClientID        string    `json:"clientID"`
    Domain          string    `json:"domain"`
    IsActive        bool      `json:"isActive"`
    LastUpdate      string    `json:"lastUpdate"`
    DatasetSize     int       `json:"datasetSize"`
    ModelAccuracy   float64   `json:"modelAccuracy"`
    DocType         string    `json:"docType"` // Added for type identification
}

// LocalModel represents a client's locally trained model
type LocalModel struct {
    ModelID         string             `json:"modelID"`
    ClientID        string             `json:"clientID"`
    Round           int                `json:"round"`
    Domain          string             `json:"domain"`
    Weights         string             `json:"weights"`
    LatentFeatures  string             `json:"latentFeatures"`
    Prototypes      string             `json:"prototypes"`
    Accuracy        float64            `json:"accuracy"`
    Loss            float64            `json:"loss"`
    AlignmentLoss   float64            `json:"alignmentLoss"`
    DataSize        int                `json:"dataSize"`
    Timestamp       string             `json:"timestamp"`
    Status          string             `json:"status"`
    DocType         string             `json:"docType"` // Added for type identification
}

// GlobalModel represents the aggregated global model
type GlobalModel struct {
    ModelID         string             `json:"modelID"`
    Version         int                `json:"version"`
    Round           int                `json:"round"`
    Weights         string             `json:"weights"`
    GlobalPrototypes string            `json:"globalPrototypes"`
    LatentDim       int                `json:"latentDim"`
    NumLatents      int                `json:"numLatents"`
    Accuracy        float64            `json:"accuracy"`
    Loss            float64            `json:"loss"`
    NumClients      int                `json:"numClients"`
    SourceClients   int                `json:"sourceClients"`
    TargetClients   int                `json:"targetClients"`
    Timestamp       string             `json:"timestamp"`
    Status          string             `json:"status"`
}

// AggregationConfig stores configuration for model aggregation
type AggregationConfig struct {
    ConfigID            string    `json:"configID"`
    MinClients          int       `json:"minClients"`
    MaxRounds           int       `json:"maxRounds"`
    SourceWeight        float64   `json:"sourceWeight"`
    TargetWeight        float64   `json:"targetWeight"`
    AlignmentWeight     float64   `json:"alignmentWeight"`
    ConvergenceThreshold float64  `json:"convergenceThreshold"`
    CurrentRound        int       `json:"currentRound"`
    LastUpdated         string    `json:"lastUpdated"`
}

// TrainingMetrics stores per-round training metrics
type TrainingMetrics struct {
    MetricID        string    `json:"metricID"`
    Round           int       `json:"round"`
    GlobalAccuracy  float64   `json:"globalAccuracy"`
    GlobalLoss      float64   `json:"globalLoss"`
    SourceAccuracy  float64   `json:"sourceAccuracy"`
    TargetAccuracy  float64   `json:"targetAccuracy"`
    AlignmentScore  float64   `json:"alignmentScore"`
    NumParticipants int       `json:"numParticipants"`
    Timestamp       string    `json:"timestamp"`
}

// getTxTimestamp retrieves the transaction timestamp
func getTxTimestamp(ctx contractapi.TransactionContextInterface) (string, error) {
    txTimestamp, err := ctx.GetStub().GetTxTimestamp()
    if err != nil {
        return "", err
    }
    return time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos)).UTC().Format(time.RFC3339Nano), nil
}

// InitLedger initializes the chaincode
func (c *VPSAContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
    timestamp, err := getTxTimestamp(ctx)
    if err != nil {
        return fmt.Errorf("failed to get transaction timestamp: %v", err)
    }

    globalModel := GlobalModel{
        ModelID:          "vpsa-global-model",
        Version:          0,
        Round:            0,
        Weights:          "{}",
        GlobalPrototypes: "{}",
        LatentDim:        768,
        NumLatents:       512,
        Accuracy:         0.0,
        Loss:             0.0,
        NumClients:       0,
        SourceClients:    0,
        TargetClients:    0,
        Timestamp:        timestamp,
        Status:           "initialized",
    }

    globalJSON, err := json.Marshal(globalModel)
    if err != nil {
        return err
    }

    err = ctx.GetStub().PutState("vpsa-global-model", globalJSON)
    if err != nil {
        return err
    }

    config := AggregationConfig{
        ConfigID:             "vpsa-config",
        MinClients:           3,
        MaxRounds:            100,
        SourceWeight:         0.6,
        TargetWeight:         0.4,
        AlignmentWeight:      0.1,
        ConvergenceThreshold: 0.001,
        CurrentRound:         0,
        LastUpdated:          timestamp,
    }

    configJSON, err := json.Marshal(config)
    if err != nil {
        return err
    }

    // Initialize client list
    err = ctx.GetStub().PutState("vpsa-config", configJSON)
    if err != nil {
        return err
    }

    // Create empty client list
    clientList := []string{}
    clientListJSON, _ := json.Marshal(clientList)
    return ctx.GetStub().PutState("client-list", clientListJSON)
}

// RegisterClient registers a new client
func (c *VPSAContract) RegisterClient(ctx contractapi.TransactionContextInterface,
    clientID string, domain string, datasetSize int) error {

    exists, err := c.ClientExists(ctx, clientID)
    if err != nil {
        return err
    }
    if exists {
        return fmt.Errorf("client %s already registered", clientID)
    }

    timestamp, err := getTxTimestamp(ctx)
    if err != nil {
        return fmt.Errorf("failed to get transaction timestamp: %v", err)
    }

    client := Client{
        ClientID:      clientID,
        Domain:        domain,
        IsActive:      true,
        LastUpdate:    timestamp,
        DatasetSize:   datasetSize,
        ModelAccuracy: 0.0,
        DocType:       "client",
    }

    clientJSON, err := json.Marshal(client)
    if err != nil {
        return err
    }

    // Store client
    err = ctx.GetStub().PutState(clientID, clientJSON)
    if err != nil {
        return err
    }

    // Update client list
    clientListJSON, err := ctx.GetStub().GetState("client-list")
    if err != nil {
        return err
    }

    var clientList []string
    if clientListJSON != nil {
        json.Unmarshal(clientListJSON, &clientList)
    }

    clientList = append(clientList, clientID)
    clientListJSON, _ = json.Marshal(clientList)
    return ctx.GetStub().PutState("client-list", clientListJSON)
}

// ClientExists checks if a client is registered
func (c *VPSAContract) ClientExists(ctx contractapi.TransactionContextInterface, clientID string) (bool, error) {
    clientJSON, err := ctx.GetStub().GetState(clientID)
    if err != nil {
        return false, fmt.Errorf("failed to read from world state: %v", err)
    }
    return clientJSON != nil, nil
}

// GetClient retrieves client information
func (c *VPSAContract) GetClient(ctx contractapi.TransactionContextInterface, clientID string) (*Client, error) {
    clientJSON, err := ctx.GetStub().GetState(clientID)
    if err != nil {
        return nil, fmt.Errorf("failed to read from world state: %v", err)
    }
    if clientJSON == nil {
        return nil, fmt.Errorf("client %s does not exist", clientID)
    }

    var client Client
    err = json.Unmarshal(clientJSON, &client)
    if err != nil {
        return nil, err
    }

    return &client, nil
}

// GetAllClients retrieves all registered clients (LevelDB compatible)
func (c *VPSAContract) GetAllClients(ctx contractapi.TransactionContextInterface) ([]*Client, error) {
    clientListJSON, err := ctx.GetStub().GetState("client-list")
    if err != nil {
        return nil, err
    }

    var clientList []string
    if clientListJSON != nil {
        json.Unmarshal(clientListJSON, &clientList)
    }

    var clients []*Client
    for _, clientID := range clientList {
        client, err := c.GetClient(ctx, clientID)
        if err == nil {
            clients = append(clients, client)
        }
    }

    return clients, nil
}

// SubmitLocalModel allows a client to submit their locally trained model
func (c *VPSAContract) SubmitLocalModel(ctx contractapi.TransactionContextInterface,
    modelID string, clientID string, weights string, latentFeatures string, 
    prototypes string, accuracy float64, loss float64, alignmentLoss float64, 
    dataSize int) error {

    client, err := c.GetClient(ctx, clientID)
    if err != nil {
        return err
    }

    if !client.IsActive {
        return fmt.Errorf("client %s is not active", clientID)
    }

    config, err := c.GetAggregationConfig(ctx)
    if err != nil {
        return err
    }

    timestamp, err := getTxTimestamp(ctx)
    if err != nil {
        return fmt.Errorf("failed to get transaction timestamp: %v", err)
    }

    localModel := LocalModel{
        ModelID:        modelID,
        ClientID:       clientID,
        Round:          config.CurrentRound,
        Domain:         client.Domain,
        Weights:        weights,
        LatentFeatures: latentFeatures,
        Prototypes:     prototypes,
        Accuracy:       accuracy,
        Loss:           loss,
        AlignmentLoss:  alignmentLoss,
        DataSize:       dataSize,
        Timestamp:      timestamp,
        Status:         "submitted",
        DocType:        "localModel",
    }

    modelJSON, err := json.Marshal(localModel)
    if err != nil {
        return err
    }

    err = ctx.GetStub().PutState(modelID, modelJSON)
    if err != nil {
        return err
    }

    // Update client
    client.LastUpdate = timestamp
    client.ModelAccuracy = accuracy
    clientJSON, err := json.Marshal(client)
    if err != nil {
        return err
    }

    // Store model in round list
    roundKey := fmt.Sprintf("round-%d-models", config.CurrentRound)
    roundModelsJSON, _ := ctx.GetStub().GetState(roundKey)
    
    var roundModels []string
    if roundModelsJSON != nil {
        json.Unmarshal(roundModelsJSON, &roundModels)
    }
    
    roundModels = append(roundModels, modelID)
    roundModelsJSON, _ = json.Marshal(roundModels)
    ctx.GetStub().PutState(roundKey, roundModelsJSON)

    return ctx.GetStub().PutState(clientID, clientJSON)
}

// GetLocalModel retrieves a local model
func (c *VPSAContract) GetLocalModel(ctx contractapi.TransactionContextInterface, modelID string) (*LocalModel, error) {
    modelJSON, err := ctx.GetStub().GetState(modelID)
    if err != nil {
        return nil, fmt.Errorf("failed to read from world state: %v", err)
    }
    if modelJSON == nil {
        return nil, fmt.Errorf("model %s does not exist", modelID)
    }

    var model LocalModel
    err = json.Unmarshal(modelJSON, &model)
    if err != nil {
        return nil, err
    }

    return &model, nil
}

// GetLocalModelsByRound retrieves all local models for a specific round (LevelDB compatible)
func (c *VPSAContract) GetLocalModelsByRound(ctx contractapi.TransactionContextInterface, 
    round int) ([]*LocalModel, error) {

    roundKey := fmt.Sprintf("round-%d-models", round)
    roundModelsJSON, err := ctx.GetStub().GetState(roundKey)
    if err != nil {
        return nil, err
    }

    var roundModels []string
    if roundModelsJSON != nil {
        json.Unmarshal(roundModelsJSON, &roundModels)
    }

    var models []*LocalModel
    for _, modelID := range roundModels {
        model, err := c.GetLocalModel(ctx, modelID)
        if err == nil && model.Status == "submitted" {
            models = append(models, model)
        }
    }

    return models, nil
}

// AggregateModels performs federated aggregation
func (c *VPSAContract) AggregateModels(ctx contractapi.TransactionContextInterface,
    modelIDs []string, aggregatedWeights string, aggregatedPrototypes string,
    globalAccuracy float64, globalLoss float64, alignmentScore float64) error {

    globalModel, err := c.GetGlobalModel(ctx)
    if err != nil {
        return fmt.Errorf("failed to get global model: %v", err)
    }

    config, err := c.GetAggregationConfig(ctx)
    if err != nil {
        return err
    }

    timestamp, err := getTxTimestamp(ctx)
    if err != nil {
        return fmt.Errorf("failed to get transaction timestamp: %v", err)
    }

    sourceCount := 0
    targetCount := 0
    
    for _, modelID := range modelIDs {
        model, err := c.GetLocalModel(ctx, modelID)
        if err != nil {
            continue
        }

        if model.Domain == "source" {
            sourceCount++
        } else if model.Domain == "target" {
            targetCount++
        }

        model.Status = "aggregated"
        modelJSON, _ := json.Marshal(model)
        ctx.GetStub().PutState(modelID, modelJSON)
    }

    globalModel.Version++
    globalModel.Round = config.CurrentRound
    globalModel.Weights = aggregatedWeights
    globalModel.GlobalPrototypes = aggregatedPrototypes
    globalModel.Accuracy = globalAccuracy
    globalModel.Loss = globalLoss
    globalModel.NumClients = len(modelIDs)
    globalModel.SourceClients = sourceCount
    globalModel.TargetClients = targetCount
    globalModel.Timestamp = timestamp
    globalModel.Status = "training"

    globalJSON, err := json.Marshal(globalModel)
    if err != nil {
        return err
    }

    err = ctx.GetStub().PutState("vpsa-global-model", globalJSON)
    if err != nil {
        return err
    }

    metrics := TrainingMetrics{
        MetricID:        fmt.Sprintf("metrics-round-%d", config.CurrentRound),
        Round:           config.CurrentRound,
        GlobalAccuracy:  globalAccuracy,
        GlobalLoss:      globalLoss,
        AlignmentScore:  alignmentScore,
        NumParticipants: len(modelIDs),
        Timestamp:       timestamp,
    }

    metricsJSON, err := json.Marshal(metrics)
    if err != nil {
        return err
    }

    err = ctx.GetStub().PutState(metrics.MetricID, metricsJSON)
    if err != nil {
        return err
    }

    config.CurrentRound++
    config.LastUpdated = timestamp
    configJSON, err := json.Marshal(config)
    if err != nil {
        return err
    }

    return ctx.GetStub().PutState("vpsa-config", configJSON)
}

// GetGlobalModel retrieves the current global model
func (c *VPSAContract) GetGlobalModel(ctx contractapi.TransactionContextInterface) (*GlobalModel, error) {
    modelJSON, err := ctx.GetStub().GetState("vpsa-global-model")
    if err != nil {
        return nil, fmt.Errorf("failed to read from world state: %v", err)
    }
    if modelJSON == nil {
        return nil, fmt.Errorf("global model does not exist")
    }

    var model GlobalModel
    err = json.Unmarshal(modelJSON, &model)
    if err != nil {
        return nil, err
    }

    return &model, nil
}

// GetAggregationConfig retrieves the aggregation configuration
func (c *VPSAContract) GetAggregationConfig(ctx contractapi.TransactionContextInterface) (*AggregationConfig, error) {
    configJSON, err := ctx.GetStub().GetState("vpsa-config")
    if err != nil {
        return nil, fmt.Errorf("failed to read config: %v", err)
    }
    if configJSON == nil {
        return nil, fmt.Errorf("config does not exist")
    }

    var config AggregationConfig
    err = json.Unmarshal(configJSON, &config)
    if err != nil {
        return nil, err
    }

    return &config, nil
}

// UpdateAggregationConfig updates aggregation parameters
func (c *VPSAContract) UpdateAggregationConfig(ctx contractapi.TransactionContextInterface,
    minClients int, sourceWeight float64, targetWeight float64, alignmentWeight float64) error {

    config, err := c.GetAggregationConfig(ctx)
    if err != nil {
        return err
    }

    timestamp, err := getTxTimestamp(ctx)
    if err != nil {
        return fmt.Errorf("failed to get transaction timestamp: %v", err)
    }

    config.MinClients = minClients
    config.SourceWeight = sourceWeight
    config.TargetWeight = targetWeight
    config.AlignmentWeight = alignmentWeight
    config.LastUpdated = timestamp

    configJSON, err := json.Marshal(config)
    if err != nil {
        return err
    }

    return ctx.GetStub().PutState("vpsa-config", configJSON)
}

// GetTrainingMetrics retrieves metrics for a specific round
func (c *VPSAContract) GetTrainingMetrics(ctx contractapi.TransactionContextInterface, 
    round int) (*TrainingMetrics, error) {

    metricID := fmt.Sprintf("metrics-round-%d", round)
    metricsJSON, err := ctx.GetStub().GetState(metricID)
    if err != nil {
        return nil, fmt.Errorf("failed to read metrics: %v", err)
    }
    if metricsJSON == nil {
        return nil, fmt.Errorf("metrics for round %d do not exist", round)
    }

    var metrics TrainingMetrics
    err = json.Unmarshal(metricsJSON, &metrics)
    if err != nil {
        return nil, err
    }

    return &metrics, nil
}

// GetAllTrainingMetrics retrieves all training metrics
func (c *VPSAContract) GetAllTrainingMetrics(ctx contractapi.TransactionContextInterface) ([]*TrainingMetrics, error) {
    config, err := c.GetAggregationConfig(ctx)
    if err != nil {
        return nil, err
    }

    var allMetrics []*TrainingMetrics
    for i := 0; i < config.CurrentRound; i++ {
        metrics, err := c.GetTrainingMetrics(ctx, i)
        if err == nil {
            allMetrics = append(allMetrics, metrics)
        }
    }

    return allMetrics, nil
}

// GetModelHistory retrieves the complete history of a model
func (c *VPSAContract) GetModelHistory(ctx contractapi.TransactionContextInterface, 
    modelID string) ([]map[string]interface{}, error) {

    resultsIterator, err := ctx.GetStub().GetHistoryForKey(modelID)
    if err != nil {
        return nil, err
    }
    defer resultsIterator.Close()

    var history []map[string]interface{}
    for resultsIterator.HasNext() {
        response, err := resultsIterator.Next()
        if err != nil {
            return nil, err
        }

        var record map[string]interface{}
        err = json.Unmarshal(response.Value, &record)
        if err != nil {
            return nil, err
        }

        record["txId"] = response.TxId
        record["timestamp"] = response.Timestamp
        record["isDelete"] = response.IsDelete

        history = append(history, record)
    }

    return history, nil
}

func main() {
    chaincode, err := contractapi.NewChaincode(&VPSAContract{})
    if err != nil {
        fmt.Printf("Error creating VPSA chaincode: %v\n", err)
        return
    }

    if err := chaincode.Start(); err != nil {
        fmt.Printf("Error starting VPSA chaincode: %v\n", err)
    }
}