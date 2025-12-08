/*
 * VPSA Chaincode with Private Data Collections
 * Virtual Prototype Semantic Alignment for Federated Learning
 *
 * Deploy with: --collections-config collections_config.json
 */

package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// Collection names
const (
	CollectionSharedModels     = "collectionSharedModels"
	CollectionAggregatedModels = "collectionAggregatedModels"
)

// VPSAContract implements the VPSA federated learning chaincode
type VPSAContract struct {
	contractapi.Contract
}

// ============================================================================
// DATA TYPES
// ============================================================================

type Client struct {
	ClientID      string  `json:"clientID"`
	Domain        string  `json:"domain"`
	IsActive      bool    `json:"isActive"`
	LastUpdate    string  `json:"lastUpdate"`
	DatasetSize   int     `json:"datasetSize"`
	ModelAccuracy float64 `json:"modelAccuracy"`
	DocType       string  `json:"docType"`
}

type LocalModelMeta struct {
	ModelID       string  `json:"modelID"`
	ClientID      string  `json:"clientID"`
	Round         int     `json:"round"`
	Domain        string  `json:"domain"`
	DataSize      int     `json:"dataSize"`
	Accuracy      float64 `json:"accuracy"`
	Loss          float64 `json:"loss"`
	AlignmentLoss float64 `json:"alignmentLoss"`
	DataHash      string  `json:"dataHash"`
	Timestamp     string  `json:"timestamp"`
	Status        string  `json:"status"`
	DocType       string  `json:"docType"`
}

type LocalModelPrivatePayload struct {
	ModelID        string               `json:"modelID"`
	ClientID       string               `json:"clientID"`
	Weights        map[string][]float64 `json:"weights"`
	LatentFeatures map[string][]float64 `json:"latentFeatures"`
	Prototypes     map[string][]float64 `json:"prototypes"`
	Accuracy       float64              `json:"accuracy"`
	Loss           float64              `json:"loss"`
	AlignmentLoss  float64              `json:"alignmentLoss"`
	DataSize       int                  `json:"dataSize"`
}

type GlobalModel struct {
	ModelID          string   `json:"modelID"`
	Version          int      `json:"version"`
	Round            int      `json:"round"`
	LatentDim        int      `json:"latentDim"`
	NumLatents       int      `json:"numLatents"`
	Accuracy         float64  `json:"accuracy"`
	Loss             float64  `json:"loss"`
	AlignmentScore   float64  `json:"alignmentScore"`
	NumClients       int      `json:"numClients"`
	SourceClients    int      `json:"sourceClients"`
	TargetClients    int      `json:"targetClients"`
	PrivateDataKey   string   `json:"privateDataKey"`
	Timestamp        string   `json:"timestamp"`
	Status           string   `json:"status"`
	ContributorsList []string `json:"contributorsList"`
}

type GlobalModelPrivatePayload struct {
	ModelID              string               `json:"modelID"`
	Round                int                  `json:"round"`
	AggregatedWeights    map[string][]float64 `json:"aggregatedWeights"`
	AggregatedPrototypes map[string][]float64 `json:"aggregatedPrototypes"`
	Timestamp            string               `json:"timestamp"`
}

type AggregationConfig struct {
	ConfigID             string  `json:"configID"`
	MinClients           int     `json:"minClients"`
	MaxRounds            int     `json:"maxRounds"`
	SourceWeight         float64 `json:"sourceWeight"`
	TargetWeight         float64 `json:"targetWeight"`
	AlignmentWeight      float64 `json:"alignmentWeight"`
	ConvergenceThreshold float64 `json:"convergenceThreshold"`
	CurrentRound         int     `json:"currentRound"`
	LastUpdated          string  `json:"lastUpdated"`
}

type TrainingMetrics struct {
	MetricID        string   `json:"metricID"`
	Round           int      `json:"round"`
	GlobalAccuracy  float64  `json:"globalAccuracy"`
	GlobalLoss      float64  `json:"globalLoss"`
	SourceAccuracy  float64  `json:"sourceAccuracy"`
	TargetAccuracy  float64  `json:"targetAccuracy"`
	AlignmentScore  float64  `json:"alignmentScore"`
	NumParticipants int      `json:"numParticipants"`
	ModelIDs        []string `json:"modelIDs"`
	Timestamp       string   `json:"timestamp"`
}

type AggregationResult struct {
	Success        bool    `json:"success"`
	NumAggregated  int     `json:"numAggregated"`
	GlobalAccuracy float64 `json:"globalAccuracy"`
	GlobalLoss     float64 `json:"globalLoss"`
	AlignmentScore float64 `json:"alignmentScore"`
	Round          int     `json:"round"`
	TxID           string  `json:"txId"`
	Message        string  `json:"message"`
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

func getTxTimestamp(ctx contractapi.TransactionContextInterface) (string, error) {
	txTimestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return "", fmt.Errorf("failed to get timestamp: %v", err)
	}
	return time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos)).UTC().Format(time.RFC3339Nano), nil
}

func readTransient(ctx contractapi.TransactionContextInterface, key string) ([]byte, error) {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return nil, fmt.Errorf("failed to get transient map: %v", err)
	}

	if data, exists := transientMap[key]; exists {
		return data, nil
	}

	for _, altKey := range []string{"model", "payload", "data"} {
		if data, ok := transientMap[altKey]; ok {
			return data, nil
		}
	}
	return nil, fmt.Errorf("key '%s' not found in transient map", key)
}

func compressGzip(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)
	if _, err := writer.Write(data); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decompressGzip(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return data, nil
	}
	defer reader.Close()
	return io.ReadAll(reader)
}

func safeAverageVectors(vectors [][]float64) ([]float64, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors")
	}
	length := len(vectors[0])
	result := make([]float64, length)
	for _, v := range vectors {
		if len(v) != length {
			return nil, fmt.Errorf("length mismatch")
		}
		for i, val := range v {
			result[i] += val
		}
	}
	for i := range result {
		result[i] /= float64(len(vectors))
	}
	return result, nil
}

func safeWeightedAverageVectors(vectors [][]float64, weights []float64) ([]float64, error) {
	if len(vectors) == 0 || len(vectors) != len(weights) {
		return nil, fmt.Errorf("invalid input")
	}
	length := len(vectors[0])
	result := make([]float64, length)
	totalWeight := 0.0
	for idx, v := range vectors {
		if len(v) != length {
			return nil, fmt.Errorf("length mismatch")
		}
		totalWeight += weights[idx]
		for i, val := range v {
			result[i] += val * weights[idx]
		}
	}
	if totalWeight > 0 {
		for i := range result {
			result[i] /= totalWeight
		}
	}
	return result, nil
}

func calculateDomainAccuracy(models []LocalModelPrivatePayload) float64 {
	if len(models) == 0 {
		return 0.0
	}
	var total float64
	var totalSize int
	for _, m := range models {
		total += m.Accuracy * float64(m.DataSize)
		totalSize += m.DataSize
	}
	if totalSize > 0 {
		return total / float64(totalSize)
	}
	return 0.0
}

// ============================================================================
// INITIALIZATION
// ============================================================================

func (c *VPSAContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	fmt.Println("INFO: InitLedger called")

	timestamp, _ := getTxTimestamp(ctx)

	globalModel := GlobalModel{
		ModelID:          "vpsa-global-model",
		Version:          0,
		Round:            0,
		LatentDim:        768,
		NumLatents:       512,
		Timestamp:        timestamp,
		Status:           "initialized",
		ContributorsList: []string{},
	}
	globalJSON, _ := json.Marshal(globalModel)
	ctx.GetStub().PutState("vpsa-global-model", globalJSON)

	config := AggregationConfig{
		ConfigID:             "vpsa-config",
		MinClients:           2,
		MaxRounds:            100,
		SourceWeight:         0.6,
		TargetWeight:         0.4,
		AlignmentWeight:      0.1,
		ConvergenceThreshold: 0.001,
		CurrentRound:         0,
		LastUpdated:          timestamp,
	}
	configJSON, _ := json.Marshal(config)
	ctx.GetStub().PutState("vpsa-config", configJSON)

	clientListJSON, _ := json.Marshal([]string{})
	ctx.GetStub().PutState("client-list", clientListJSON)

	fmt.Println("INFO: InitLedger completed")
	return nil
}

// ============================================================================
// CLIENT MANAGEMENT
// ============================================================================

func (c *VPSAContract) RegisterClient(ctx contractapi.TransactionContextInterface,
	clientID string, domain string, datasetSize int) error {

	fmt.Printf("INFO: RegisterClient - %s, %s\n", clientID, domain)

	if domain != "source" && domain != "target" {
		return fmt.Errorf("invalid domain: must be 'source' or 'target'")
	}

	exists, _ := c.ClientExists(ctx, clientID)
	if exists {
		return fmt.Errorf("client %s already exists", clientID)
	}

	timestamp, _ := getTxTimestamp(ctx)

	client := Client{
		ClientID:    clientID,
		Domain:      domain,
		IsActive:    true,
		LastUpdate:  timestamp,
		DatasetSize: datasetSize,
		DocType:     "client",
	}
	clientJSON, _ := json.Marshal(client)
	ctx.GetStub().PutState(clientID, clientJSON)

	clientListJSON, _ := ctx.GetStub().GetState("client-list")
	var clientList []string
	json.Unmarshal(clientListJSON, &clientList)
	clientList = append(clientList, clientID)
	clientListJSON, _ = json.Marshal(clientList)
	ctx.GetStub().PutState("client-list", clientListJSON)

	return nil
}

func (c *VPSAContract) ClientExists(ctx contractapi.TransactionContextInterface, clientID string) (bool, error) {
	data, err := ctx.GetStub().GetState(clientID)
	return data != nil, err
}

func (c *VPSAContract) GetClient(ctx contractapi.TransactionContextInterface, clientID string) (*Client, error) {
	data, err := ctx.GetStub().GetState(clientID)
	if err != nil || data == nil {
		return nil, fmt.Errorf("client %s not found", clientID)
	}
	var client Client
	json.Unmarshal(data, &client)
	return &client, nil
}

func (c *VPSAContract) GetAllClients(ctx contractapi.TransactionContextInterface) ([]*Client, error) {
	listJSON, _ := ctx.GetStub().GetState("client-list")
	var list []string
	json.Unmarshal(listJSON, &list)

	var clients []*Client
	for _, id := range list {
		if client, err := c.GetClient(ctx, id); err == nil {
			clients = append(clients, client)
		}
	}
	return clients, nil
}

// ============================================================================
// MODEL SUBMISSION
// ============================================================================

func (c *VPSAContract) SubmitLocalModel(ctx contractapi.TransactionContextInterface,
	modelID, clientID, weights, latentFeatures, prototypes string,
	accuracy, loss, alignmentLoss float64, dataSize int) error {

	fmt.Printf("INFO: SubmitLocalModel - %s\n", modelID)

	client, err := c.GetClient(ctx, clientID)
	if err != nil {
		return err
	}
	if !client.IsActive {
		return fmt.Errorf("client %s inactive", clientID)
	}

	config, _ := c.GetAggregationConfig(ctx)
	timestamp, _ := getTxTimestamp(ctx)

	// Parse and store in private collection
	var weightsMap, latentMap, protoMap map[string][]float64
	json.Unmarshal([]byte(weights), &weightsMap)
	json.Unmarshal([]byte(latentFeatures), &latentMap)
	json.Unmarshal([]byte(prototypes), &protoMap)

	privatePayload := LocalModelPrivatePayload{
		ModelID:        modelID,
		ClientID:       clientID,
		Weights:        weightsMap,
		LatentFeatures: latentMap,
		Prototypes:     protoMap,
		Accuracy:       accuracy,
		Loss:           loss,
		AlignmentLoss:  alignmentLoss,
		DataSize:       dataSize,
	}
	payloadJSON, _ := json.Marshal(privatePayload)
	ctx.GetStub().PutPrivateData(CollectionSharedModels, modelID, payloadJSON)

	// Store public metadata
	meta := LocalModelMeta{
		ModelID:       modelID,
		ClientID:      clientID,
		Round:         config.CurrentRound,
		Domain:        client.Domain,
		DataSize:      dataSize,
		Accuracy:      accuracy,
		Loss:          loss,
		AlignmentLoss: alignmentLoss,
		Timestamp:     timestamp,
		Status:        "submitted",
		DocType:       "localModel",
	}
	metaJSON, _ := json.Marshal(meta)
	ctx.GetStub().PutState(modelID, metaJSON)

	// Update round list
	roundKey := fmt.Sprintf("round-%d-models", config.CurrentRound)
	roundJSON, _ := ctx.GetStub().GetState(roundKey)
	var roundModels []string
	json.Unmarshal(roundJSON, &roundModels)
	roundModels = append(roundModels, modelID)
	roundJSON, _ = json.Marshal(roundModels)
	ctx.GetStub().PutState(roundKey, roundJSON)

	// Update client
	client.LastUpdate = timestamp
	client.ModelAccuracy = accuracy
	clientJSON, _ := json.Marshal(client)
	ctx.GetStub().PutState(clientID, clientJSON)

	return nil
}

func (c *VPSAContract) SubmitLocalModelWithTransient(ctx contractapi.TransactionContextInterface,
	modelID, clientID string, round int, dataHash string) (string, error) {

	fmt.Printf("INFO: SubmitLocalModelWithTransient - %s\n", modelID)

	client, err := c.GetClient(ctx, clientID)
	if err != nil {
		return "", err
	}

	transientData, err := readTransient(ctx, "model")
	if err != nil {
		return "", err
	}

	decompressed, _ := decompressGzip(transientData)

	var payload LocalModelPrivatePayload
	if err := json.Unmarshal(decompressed, &payload); err != nil {
		return "", err
	}

	payload.ModelID = modelID
	payload.ClientID = clientID

	payloadJSON, _ := json.Marshal(payload)
	compressed, _ := compressGzip(payloadJSON)
	ctx.GetStub().PutPrivateData(CollectionSharedModels, modelID, compressed)

	timestamp, _ := getTxTimestamp(ctx)

	meta := LocalModelMeta{
		ModelID:       modelID,
		ClientID:      clientID,
		Round:         round,
		Domain:        client.Domain,
		DataSize:      payload.DataSize,
		Accuracy:      payload.Accuracy,
		Loss:          payload.Loss,
		AlignmentLoss: payload.AlignmentLoss,
		DataHash:      dataHash,
		Timestamp:     timestamp,
		Status:        "submitted",
		DocType:       "localModelMeta",
	}
	metaJSON, _ := json.Marshal(meta)
	ctx.GetStub().PutState(modelID+"-meta", metaJSON)

	roundKey := fmt.Sprintf("round-%d-models", round)
	roundJSON, _ := ctx.GetStub().GetState(roundKey)
	var roundModels []string
	json.Unmarshal(roundJSON, &roundModels)
	roundModels = append(roundModels, modelID)
	roundJSON, _ = json.Marshal(roundModels)
	ctx.GetStub().PutState(roundKey, roundJSON)

	client.LastUpdate = timestamp
	client.ModelAccuracy = payload.Accuracy
	clientJSON, _ := json.Marshal(client)
	ctx.GetStub().PutState(clientID, clientJSON)

	return timestamp, nil
}

func (c *VPSAContract) GetLocalModel(ctx contractapi.TransactionContextInterface, modelID string) (*LocalModelMeta, error) {
	data, err := ctx.GetStub().GetState(modelID)
	if err != nil || data == nil {
		return nil, fmt.Errorf("model %s not found", modelID)
	}
	var model LocalModelMeta
	json.Unmarshal(data, &model)
	return &model, nil
}

func (c *VPSAContract) GetLocalModelsByRound(ctx contractapi.TransactionContextInterface, round int) ([]*LocalModelMeta, error) {
	roundKey := fmt.Sprintf("round-%d-models", round)
	roundJSON, _ := ctx.GetStub().GetState(roundKey)
	var roundModels []string
	json.Unmarshal(roundJSON, &roundModels)

	var models []*LocalModelMeta
	for _, id := range roundModels {
		if m, err := c.GetLocalModel(ctx, id); err == nil && m.Status == "submitted" {
			models = append(models, m)
		}
	}
	return models, nil
}

// ============================================================================
// AGGREGATION
// ============================================================================

func (c *VPSAContract) AggregateModelsWithPrivateData(ctx contractapi.TransactionContextInterface,
	modelIDsJSON string) (*AggregationResult, error) {

	fmt.Println("INFO: AggregateModelsWithPrivateData called")

	txID := ctx.GetStub().GetTxID()

	var modelIDs []string
	json.Unmarshal([]byte(modelIDsJSON), &modelIDs)

	if len(modelIDs) == 0 {
		return nil, fmt.Errorf("no model IDs")
	}

	config, _ := c.GetAggregationConfig(ctx)

	if len(modelIDs) < config.MinClients {
		return &AggregationResult{
			Success: false,
			Message: fmt.Sprintf("need %d models, got %d", config.MinClients, len(modelIDs)),
			TxID:    txID,
		}, nil
	}

	var sourceModels, targetModels, allModels []LocalModelPrivatePayload
	var processedIDs, skippedIDs []string

	for _, modelID := range modelIDs {
		privateData, _ := ctx.GetStub().GetPrivateData(CollectionSharedModels, modelID)

		var payload LocalModelPrivatePayload

		if privateData == nil {
			metaJSON, _ := ctx.GetStub().GetState(modelID)
			if metaJSON == nil {
				skippedIDs = append(skippedIDs, modelID)
				continue
			}
			var meta LocalModelMeta
			json.Unmarshal(metaJSON, &meta)
			payload = LocalModelPrivatePayload{
				ModelID:    meta.ModelID,
				ClientID:   meta.ClientID,
				Accuracy:   meta.Accuracy,
				Loss:       meta.Loss,
				DataSize:   meta.DataSize,
				Weights:    make(map[string][]float64),
				Prototypes: make(map[string][]float64),
			}
		} else {
			decompressed, _ := decompressGzip(privateData)
			json.Unmarshal(decompressed, &payload)
		}

		client, _ := c.GetClient(ctx, payload.ClientID)
		domain := "unknown"
		if client != nil {
			domain = client.Domain
		}

		if domain == "source" {
			sourceModels = append(sourceModels, payload)
		} else {
			targetModels = append(targetModels, payload)
		}
		allModels = append(allModels, payload)
		processedIDs = append(processedIDs, modelID)
	}

	if len(allModels) == 0 {
		return &AggregationResult{Success: false, Message: "no valid models", TxID: txID}, nil
	}

	// VPSA Aggregation
	aggregatedWeights := make(map[string][]float64)
	aggregatedPrototypes := make(map[string][]float64)

	allWeightKeys := make(map[string]bool)
	for _, m := range allModels {
		for k := range m.Weights {
			allWeightKeys[k] = true
		}
	}

	for key := range allWeightKeys {
		var srcVecs [][]float64
		var srcW []float64
		var tgtVecs [][]float64
		var tgtW []float64

		for _, m := range sourceModels {
			if v, ok := m.Weights[key]; ok && len(v) > 0 {
				srcVecs = append(srcVecs, v)
				srcW = append(srcW, float64(m.DataSize))
			}
		}
		for _, m := range targetModels {
			if v, ok := m.Weights[key]; ok && len(v) > 0 {
				tgtVecs = append(tgtVecs, v)
				tgtW = append(tgtW, float64(m.DataSize))
			}
		}

		var srcAvg, tgtAvg []float64
		if len(srcVecs) > 0 {
			srcAvg, _ = safeWeightedAverageVectors(srcVecs, srcW)
		}
		if len(tgtVecs) > 0 {
			tgtAvg, _ = safeWeightedAverageVectors(tgtVecs, tgtW)
		}

		if srcAvg != nil && tgtAvg != nil && len(srcAvg) == len(tgtAvg) {
			combined := make([]float64, len(srcAvg))
			for i := range combined {
				combined[i] = srcAvg[i]*config.SourceWeight + tgtAvg[i]*config.TargetWeight
			}
			aggregatedWeights[key] = combined
		} else if srcAvg != nil {
			aggregatedWeights[key] = srcAvg
		} else if tgtAvg != nil {
			aggregatedWeights[key] = tgtAvg
		}
	}

	// Aggregate prototypes
	allProtoKeys := make(map[string]bool)
	for _, m := range allModels {
		for k := range m.Prototypes {
			allProtoKeys[k] = true
		}
	}
	for key := range allProtoKeys {
		var vecs [][]float64
		for _, m := range allModels {
			if v, ok := m.Prototypes[key]; ok && len(v) > 0 {
				vecs = append(vecs, v)
			}
		}
		if avg, err := safeAverageVectors(vecs); err == nil {
			aggregatedPrototypes[key] = avg
		}
	}

	// Compute metrics
	var totalAcc, totalLoss, totalAlign float64
	var totalSize int
	for _, m := range allModels {
		totalAcc += m.Accuracy * float64(m.DataSize)
		totalLoss += m.Loss * float64(m.DataSize)
		totalAlign += m.AlignmentLoss * float64(m.DataSize)
		totalSize += m.DataSize
	}

	globalAccuracy := totalAcc / float64(totalSize)
	globalLoss := totalLoss / float64(totalSize)
	alignmentScore := 1.0 - (totalAlign / float64(totalSize))

	timestamp, _ := getTxTimestamp(ctx)

	// Store aggregated in private
	privateKey := fmt.Sprintf("vpsa-global-round-%d", config.CurrentRound)
	globalPrivate := GlobalModelPrivatePayload{
		ModelID:              privateKey,
		Round:                config.CurrentRound,
		AggregatedWeights:    aggregatedWeights,
		AggregatedPrototypes: aggregatedPrototypes,
		Timestamp:            timestamp,
	}
	privateJSON, _ := json.Marshal(globalPrivate)
	compressed, _ := compressGzip(privateJSON)
	ctx.GetStub().PutPrivateData(CollectionAggregatedModels, privateKey, compressed)

	// Update public global model
	globalModel := GlobalModel{
		ModelID:          "vpsa-global-model",
		Version:          config.CurrentRound + 1,
		Round:            config.CurrentRound,
		Accuracy:         globalAccuracy,
		Loss:             globalLoss,
		AlignmentScore:   alignmentScore,
		NumClients:       len(allModels),
		SourceClients:    len(sourceModels),
		TargetClients:    len(targetModels),
		PrivateDataKey:   privateKey,
		Timestamp:        timestamp,
		Status:           "aggregated",
		ContributorsList: processedIDs,
	}
	globalJSON, _ := json.Marshal(globalModel)
	ctx.GetStub().PutState("vpsa-global-model", globalJSON)

	// Mark models aggregated
	for _, id := range processedIDs {
		if data, _ := ctx.GetStub().GetState(id); data != nil {
			var meta LocalModelMeta
			json.Unmarshal(data, &meta)
			meta.Status = "aggregated"
			updated, _ := json.Marshal(meta)
			ctx.GetStub().PutState(id, updated)
		}
	}

	// Store metrics
	metrics := TrainingMetrics{
		MetricID:        fmt.Sprintf("metrics-round-%d", config.CurrentRound),
		Round:           config.CurrentRound,
		GlobalAccuracy:  globalAccuracy,
		GlobalLoss:      globalLoss,
		SourceAccuracy:  calculateDomainAccuracy(sourceModels),
		TargetAccuracy:  calculateDomainAccuracy(targetModels),
		AlignmentScore:  alignmentScore,
		NumParticipants: len(allModels),
		ModelIDs:        processedIDs,
		Timestamp:       timestamp,
	}
	metricsJSON, _ := json.Marshal(metrics)
	ctx.GetStub().PutState(metrics.MetricID, metricsJSON)

	// Increment round
	config.CurrentRound++
	config.LastUpdated = timestamp
	configJSON, _ := json.Marshal(config)
	ctx.GetStub().PutState("vpsa-config", configJSON)

	return &AggregationResult{
		Success:        true,
		NumAggregated:  len(allModels),
		GlobalAccuracy: globalAccuracy,
		GlobalLoss:     globalLoss,
		AlignmentScore: alignmentScore,
		Round:          config.CurrentRound - 1,
		TxID:           txID,
		Message:        fmt.Sprintf("Aggregated %d (skipped %d)", len(allModels), len(skippedIDs)),
	}, nil
}

// ============================================================================
// QUERY FUNCTIONS
// ============================================================================

func (c *VPSAContract) GetGlobalModel(ctx contractapi.TransactionContextInterface) (*GlobalModel, error) {
	data, _ := ctx.GetStub().GetState("vpsa-global-model")
	if data == nil {
		return nil, fmt.Errorf("global model not found")
	}
	var model GlobalModel
	json.Unmarshal(data, &model)
	return &model, nil
}

func (c *VPSAContract) GetAggregationConfig(ctx contractapi.TransactionContextInterface) (*AggregationConfig, error) {
	data, _ := ctx.GetStub().GetState("vpsa-config")
	if data == nil {
		return nil, fmt.Errorf("config not found")
	}
	var config AggregationConfig
	json.Unmarshal(data, &config)
	return &config, nil
}

func (c *VPSAContract) UpdateAggregationConfig(ctx contractapi.TransactionContextInterface,
	minClients int, sourceWeight, targetWeight, alignmentWeight float64) error {

	config, _ := c.GetAggregationConfig(ctx)
	timestamp, _ := getTxTimestamp(ctx)

	config.MinClients = minClients
	config.SourceWeight = sourceWeight
	config.TargetWeight = targetWeight
	config.AlignmentWeight = alignmentWeight
	config.LastUpdated = timestamp

	configJSON, _ := json.Marshal(config)
	return ctx.GetStub().PutState("vpsa-config", configJSON)
}

func (c *VPSAContract) GetTrainingMetrics(ctx contractapi.TransactionContextInterface, round int) (*TrainingMetrics, error) {
	data, _ := ctx.GetStub().GetState(fmt.Sprintf("metrics-round-%d", round))
	if data == nil {
		return nil, fmt.Errorf("metrics not found")
	}
	var metrics TrainingMetrics
	json.Unmarshal(data, &metrics)
	return &metrics, nil
}

func (c *VPSAContract) GetAllTrainingMetrics(ctx contractapi.TransactionContextInterface) ([]*TrainingMetrics, error) {
	config, _ := c.GetAggregationConfig(ctx)
	var all []*TrainingMetrics
	for i := 0; i < config.CurrentRound; i++ {
		if m, err := c.GetTrainingMetrics(ctx, i); err == nil {
			all = append(all, m)
		}
	}
	return all, nil
}

func (c *VPSAContract) GetModelHistory(ctx contractapi.TransactionContextInterface, modelID string) ([]map[string]interface{}, error) {
	iter, err := ctx.GetStub().GetHistoryForKey(modelID)
	if err != nil {
		return nil, err
	}
	defer iter.Close()

	var history []map[string]interface{}
	for iter.HasNext() {
		resp, _ := iter.Next()
		var record map[string]interface{}
		json.Unmarshal(resp.Value, &record)
		record["txId"] = resp.TxId
		record["timestamp"] = resp.Timestamp
		history = append(history, record)
	}
	return history, nil
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	chaincode, err := contractapi.NewChaincode(&VPSAContract{})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}
