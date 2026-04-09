package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ---------- Data types ----------

type VPSAContract struct {
	contractapi.Contract
}

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
	ModelID  string `json:"modelID"`
	ClientID string `json:"clientID"`
	Round    int    `json:"round"`
	DocType  string `json:"docType"`
	Status   string `json:"status"`
	// partitions are not stored in public state; metadata only
}

type GlobalModel struct {
	ModelID          string  `json:"modelID"`
	Version          int     `json:"version"`
	Round            int     `json:"round"`
	Weights          string  `json:"weights"` // returned at query time; not persisted in world state
	WeightsHash      string  `json:"weightsHash,omitempty"`
	GlobalPrototypes string  `json:"globalPrototypes"`
	LatentDim        int     `json:"latentDim"`
	NumLatents       int     `json:"numLatents"`
	Accuracy         float64 `json:"accuracy"`
	Loss             float64 `json:"loss"`
	NumClients       int     `json:"numClients"`
	SourceClients    int     `json:"sourceClients"`
	TargetClients    int     `json:"targetClients"`
	Timestamp        string  `json:"timestamp"`
	Status           string  `json:"status"`
}

type AggregationConfig struct {
	ConfigID             string   `json:"configID"`
	MinClients           int      `json:"minClients"`
	MaxRounds            int      `json:"maxRounds"`
	SourceWeight         float64  `json:"sourceWeight"`
	TargetWeight         float64  `json:"targetWeight"`
	AlignmentWeight      float64  `json:"alignmentWeight"`
	ConvergenceThreshold float64  `json:"convergenceThreshold"`
	CurrentRound         int      `json:"currentRound"`
	LastUpdated          string   `json:"lastUpdated"`
	Collections          []string `json:"collections"` // private collections used for partitions
	Beta                 int      `json:"beta"`        // num outliers to trim per coordinate
	Alpha                float64  `json:"alpha"`       // anomaly score weighting: α*dist + (1-α)*(1-cos)
}

type TrainingMetrics struct {
	MetricID        string  `json:"metricID"`
	Round           int     `json:"round"`
	GlobalAccuracy  float64 `json:"globalAccuracy"`
	GlobalLoss      float64 `json:"globalLoss"`
	SourceAccuracy  float64 `json:"sourceAccuracy"`
	TargetAccuracy  float64 `json:"targetAccuracy"`
	AlignmentScore  float64 `json:"alignmentScore"`
	NumParticipants int     `json:"numParticipants"`
	Timestamp       string  `json:"timestamp"`
}

const (
	globalModelWeightsCollection = "collectionGlobalModelPrivate"
	globalModelWeightsKey        = "vpsa-global-model::weights"
)

// ---------- Trust Score Types ----------

const TrustDecayGamma = 0.8 // EMA decay: how much history matters (higher = more memory)

// TrustRoundEntry records one round's trust contribution for a client
type TrustRoundEntry struct {
	Round        int     `json:"round"`
	AnomalyScore float64 `json:"anomalyScore"`
	WasRejected  bool    `json:"wasRejected"`
	TrustBefore  float64 `json:"trustBefore"`
	TrustAfter   float64 `json:"trustAfter"`
}

// ClientTrustScore stores accumulated trust across rounds for one client
type ClientTrustScore struct {
	ClientID           string            `json:"clientID"`
	TrustScore         float64           `json:"trustScore"`         // Current EMA trust [0,1]
	RoundsParticipated int               `json:"roundsParticipated"` // Total rounds participated
	RoundsRejected     int               `json:"roundsRejected"`     // How many times rejected
	LastRound          int               `json:"lastRound"`          // Last round participated
	History            []TrustRoundEntry `json:"history"`            // Per-round trust breakdown
	DocType            string            `json:"docType"`
}

// ---------- Helpers ----------

func getTxTimestamp(ctx contractapi.TransactionContextInterface) (string, error) {
	txTimestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return "", err
	}
	return time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos)).UTC().Format(time.RFC3339Nano), nil
}

func floatSliceFromJSONBytes(b []byte) ([]float64, error) {
	var v []float64
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, err
	}
	return v, nil
}

func jsonBytesFromFloatSlice(v []float64) ([]byte, error) {
	return json.Marshal(v)
}

func hashBytesSHA256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func (c *VPSAContract) loadGlobalWeightsPrivate(ctx contractapi.TransactionContextInterface, gm *GlobalModel) ([]float64, error) {
	privateBytes, err := ctx.GetStub().GetPrivateData(globalModelWeightsCollection, globalModelWeightsKey)
	if err != nil {
		return nil, fmt.Errorf("read global weights private data failed: %v", err)
	}
	if privateBytes != nil {
		weights, err := floatSliceFromJSONBytes(privateBytes)
		if err != nil {
			return nil, fmt.Errorf("invalid private global weights json: %v", err)
		}
		return weights, nil
	}

	// Backward compatibility: fall back to world-state encoded weights if present.
	if gm != nil && strings.TrimSpace(gm.Weights) != "" {
		weights, err := floatSliceFromJSONBytes([]byte(gm.Weights))
		if err == nil {
			return weights, nil
		}
	}

	return []float64{}, nil
}

func (c *VPSAContract) storeGlobalWeightsPrivate(ctx contractapi.TransactionContextInterface, weights []float64) (string, error) {
	weightsBytes, err := jsonBytesFromFloatSlice(weights)
	if err != nil {
		return "", fmt.Errorf("marshal global weights failed: %v", err)
	}

	if err := ctx.GetStub().PutPrivateData(globalModelWeightsCollection, globalModelWeightsKey, weightsBytes); err != nil {
		return "", fmt.Errorf("put private global weights failed: %v", err)
	}

	return hashBytesSHA256Hex(weightsBytes), nil
}

// Euclidean norm
func norm(vec []float64) float64 {
	var s float64
	for _, x := range vec {
		s += x * x
	}
	return math.Sqrt(s)
}

// dot product
func dot(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("dot: length mismatch")
	}
	var s float64
	for i := 0; i < len(a); i++ {
		s += a[i] * b[i]
	}
	return s, nil
}

// cosine similarity between a and b
func cosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("cosineSimilarity: length mismatch")
	}
	adotb, err := dot(a, b)
	if err != nil {
		return 0, err
	}
	na := norm(a)
	nb := norm(b)
	if na == 0 || nb == 0 {
		return 0, errors.New("cosineSimilarity: zero norm")
	}
	return adotb / (na * nb), nil
}

// safeDiv avoids NaN
func safeDiv(num, den float64) float64 {
	if math.Abs(den) < 1e-12 {
		return 0.0
	}
	return num / den
}

// PAPER-EXACT: Modulo-based vertical partitioning (Algorithm 1)
// Paper rule: position j goes to collection ((totalSize - 1 - j) % numCollections)
// This is interleaved partitioning from the last index, as specified in VPSA paper.
func paperPartitionIndices(totalSize, numCollections int) [][]int {
	partitions := make([][]int, numCollections)
	for i := 0; i < numCollections; i++ {
		partitions[i] = []int{}
	}
	for j := 0; j < totalSize; j++ {
		collIdx := (totalSize - 1 - j) % numCollections
		partitions[collIdx] = append(partitions[collIdx], j)
	}
	return partitions
}

// extractByIndices extracts elements from a full vector at the given positions
func extractByIndices(full []float64, indices []int) []float64 {
	result := make([]float64, len(indices))
	for i, idx := range indices {
		if idx < len(full) {
			result[i] = full[idx]
		}
	}
	return result
}

// scatterByIndices places values into a full vector at the given positions
func scatterByIndices(full []float64, indices []int, values []float64) {
	for i, idx := range indices {
		if idx < len(full) && i < len(values) {
			full[idx] = values[i]
		}
	}
}

// medianFloat64 computes the median of a float64 slice
func medianFloat64(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// percentileFloat64 computes the p-th percentile using linear interpolation
func percentileFloat64(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	n := len(sorted)
	if n == 1 {
		return sorted[0]
	}
	idx := p / 100.0 * float64(n-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	if lower == upper || upper >= n {
		return sorted[lower]
	}
	frac := idx - float64(lower)
	return sorted[lower]*(1-frac) + sorted[upper]*frac
}

// ---------- Init / Register / Getters ----------

func (c *VPSAContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	timestamp, err := getTxTimestamp(ctx)
	if err != nil {
		return err
	}

	global := GlobalModel{
		ModelID:          "vpsa-global-model",
		Version:          0,
		Round:            0,
		Weights:          "",
		WeightsHash:      hashBytesSHA256Hex([]byte("[]")),
		GlobalPrototypes: "[]",
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

	b, _ := json.Marshal(global)
	if err := ctx.GetStub().PutState("vpsa-global-model", b); err != nil {
		return err
	}

	// default collections (example). Edit to match collections_config.json in your network.
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
		Collections:          []string{"collectionOrg1Private", "collectionOrg2Private"},
		Beta:                 1,
		Alpha:                0.5,
	}

	cj, _ := json.Marshal(config)
	if err := ctx.GetStub().PutState("vpsa-config", cj); err != nil {
		return err
	}

	// empty client list
	clientList := []string{}
	clientListJSON, _ := json.Marshal(clientList)
	return ctx.GetStub().PutState("client-list", clientListJSON)
}

func (c *VPSAContract) RegisterClient(ctx contractapi.TransactionContextInterface, clientID string, domain string, datasetSize int) error {
	exists, err := c.ClientExists(ctx, clientID)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("client %s already registered", clientID)
	}

	timestamp, err := getTxTimestamp(ctx)
	if err != nil {
		return err
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
	clientJSON, _ := json.Marshal(client)
	if err := ctx.GetStub().PutState(clientID, clientJSON); err != nil {
		return err
	}

	// update client-list
	clientListJSON, err := ctx.GetStub().GetState("client-list")
	if err != nil {
		return err
	}
	var clientList []string
	if clientListJSON != nil {
		_ = json.Unmarshal(clientListJSON, &clientList)
	}
	clientList = append(clientList, clientID)
	clientListJSON, _ = json.Marshal(clientList)
	return ctx.GetStub().PutState("client-list", clientListJSON)
}

func (c *VPSAContract) ClientExists(ctx contractapi.TransactionContextInterface, clientID string) (bool, error) {
	clientJSON, err := ctx.GetStub().GetState(clientID)
	if err != nil {
		return false, err
	}
	return clientJSON != nil, nil
}

func (c *VPSAContract) GetAggregationConfig(ctx contractapi.TransactionContextInterface) (*AggregationConfig, error) {
	configJSON, err := ctx.GetStub().GetState("vpsa-config")
	if err != nil {
		return nil, err
	}
	if configJSON == nil {
		return nil, fmt.Errorf("config does not exist")
	}
	var cfg AggregationConfig
	if err := json.Unmarshal(configJSON, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *VPSAContract) GetGlobalModel(ctx contractapi.TransactionContextInterface) (*GlobalModel, error) {
	modelJSON, err := ctx.GetStub().GetState("vpsa-global-model")
	if err != nil {
		return nil, err
	}
	if modelJSON == nil {
		return nil, fmt.Errorf("global model missing")
	}
	var gm GlobalModel
	if err := json.Unmarshal(modelJSON, &gm); err != nil {
		return nil, err
	}

	weights, err := c.loadGlobalWeightsPrivate(ctx, &gm)
	if err != nil {
		return nil, err
	}
	if wb, err := json.Marshal(weights); err == nil {
		gm.Weights = string(wb)
	}
	if gm.WeightsHash == "" {
		gm.WeightsHash = hashBytesSHA256Hex([]byte(gm.Weights))
	}
	return &gm, nil
}

func (c *VPSAContract) SetGlobalModelWeights(ctx contractapi.TransactionContextInterface, weightsJSON string) error {
	// Get current global model
	gm, err := c.GetGlobalModel(ctx)
	if err != nil {
		return err
	}

	// Validate weights can be parsed
	var weights []float64
	if err := json.Unmarshal([]byte(weightsJSON), &weights); err != nil {
		return fmt.Errorf("invalid weights JSON: %v", err)
	}

	weightsHash, err := c.storeGlobalWeightsPrivate(ctx, weights)
	if err != nil {
		return err
	}

	// Keep only metadata/hash in world state.
	gm.Weights = ""
	gm.WeightsHash = weightsHash
	gm.Version++
	timestamp, _ := getTxTimestamp(ctx)
	gm.Timestamp = timestamp
	gm.Status = "initialized"

	// Save updated model
	gmBytes, err := json.Marshal(gm)
	if err != nil {
		return fmt.Errorf("marshal global model failed: %v", err)
	}

	if err := ctx.GetStub().PutState("vpsa-global-model", gmBytes); err != nil {
		return fmt.Errorf("put global model failed: %v", err)
	}

	return nil
}

// SetGlobalModelWeightsTransient uses transient data to avoid argument length limits
func (c *VPSAContract) SetGlobalModelWeightsTransient(ctx contractapi.TransactionContextInterface) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("failed to get transient: %v", err)
	}

	weightsBase64, ok := transientMap["globalWeights"]
	if !ok {
		return fmt.Errorf("transient map must contain 'globalWeights'")
	}

	// Decode base64
	weightsJSON, err := base64.StdEncoding.DecodeString(string(weightsBase64))
	if err != nil {
		return fmt.Errorf("failed to decode base64 weights: %v", err)
	}

	// Parse weights
	var weights []float64
	if err := json.Unmarshal(weightsJSON, &weights); err != nil {
		return fmt.Errorf("invalid weights JSON: %v", err)
	}

	// Get current global model
	gm, err := c.GetGlobalModel(ctx)
	if err != nil {
		return err
	}

	weightsHash, err := c.storeGlobalWeightsPrivate(ctx, weights)
	if err != nil {
		return err
	}

	// Keep only metadata/hash in world state.
	gm.Weights = ""
	gm.WeightsHash = weightsHash
	gm.Version++
	timestamp, _ := getTxTimestamp(ctx)
	gm.Timestamp = timestamp
	gm.Status = "initialized"

	// Save updated model
	gmBytes, err := json.Marshal(gm)
	if err != nil {
		return fmt.Errorf("marshal global model failed: %v", err)
	}

	if err := ctx.GetStub().PutState("vpsa-global-model", gmBytes); err != nil {
		return fmt.Errorf("put global model failed: %v", err)
	}

	return nil
}

// SetGlobalModelWeightsChunkStart initiates a chunked upload session for large models
func (c *VPSAContract) SetGlobalModelWeightsChunkStart(ctx contractapi.TransactionContextInterface, totalChunks int, totalParams int) error {
	chunkState := map[string]interface{}{
		"totalChunks":    totalChunks,
		"totalParams":    totalParams,
		"receivedChunks": 0,
		"startTime":      time.Now().Unix(),
	}

	stateBytes, err := json.Marshal(chunkState)
	if err != nil {
		return fmt.Errorf("failed to marshal chunk state: %v", err)
	}

	if err := ctx.GetStub().PutState("vpsa-weights-chunked-upload", stateBytes); err != nil {
		return fmt.Errorf("failed to put chunk state: %v", err)
	}

	return nil
}

// SetGlobalModelWeightsChunk uploads a single chunk of weights
func (c *VPSAContract) SetGlobalModelWeightsChunk(ctx contractapi.TransactionContextInterface, chunkIndex int, weightsJSON string) error {
	// Get chunk state
	stateBytes, err := ctx.GetStub().GetState("vpsa-weights-chunked-upload")
	if err != nil {
		return fmt.Errorf("failed to get chunk state: %v", err)
	}
	if stateBytes == nil {
		return fmt.Errorf("no active chunked upload session; call SetGlobalModelWeightsChunkStart first")
	}

	var chunkState map[string]interface{}
	if err := json.Unmarshal(stateBytes, &chunkState); err != nil {
		return fmt.Errorf("failed to unmarshal chunk state: %v", err)
	}

	// Validate chunk
	var weights []float64
	if err := json.Unmarshal([]byte(weightsJSON), &weights); err != nil {
		return fmt.Errorf("invalid weights JSON in chunk: %v", err)
	}

	// Store chunk data privately to avoid public ledger bloat.
	chunkKey := fmt.Sprintf("vpsa-weight-chunk-%d", chunkIndex)
	if err := ctx.GetStub().PutPrivateData(globalModelWeightsCollection, chunkKey, []byte(weightsJSON)); err != nil {
		return fmt.Errorf("failed to store chunk: %v", err)
	}

	// Update chunk state
	chunkState["receivedChunks"] = int(chunkState["receivedChunks"].(float64)) + 1

	stateBytes, err = json.Marshal(chunkState)
	if err != nil {
		return fmt.Errorf("failed to marshal updated chunk state: %v", err)
	}

	if err := ctx.GetStub().PutState("vpsa-weights-chunked-upload", stateBytes); err != nil {
		return fmt.Errorf("failed to update chunk state: %v", err)
	}

	return nil
}

// SetGlobalModelWeightsChunkFinalize assembles all chunks and updates the global model
func (c *VPSAContract) SetGlobalModelWeightsChunkFinalize(ctx contractapi.TransactionContextInterface) error {
	// Get chunk state
	stateBytes, err := ctx.GetStub().GetState("vpsa-weights-chunked-upload")
	if err != nil {
		return fmt.Errorf("failed to get chunk state: %v", err)
	}
	if stateBytes == nil {
		return fmt.Errorf("no active chunked upload session")
	}

	var chunkState map[string]interface{}
	if err := json.Unmarshal(stateBytes, &chunkState); err != nil {
		return fmt.Errorf("failed to unmarshal chunk state: %v", err)
	}

	totalChunks := int(chunkState["totalChunks"].(float64))
	receivedChunks := int(chunkState["receivedChunks"].(float64))

	if receivedChunks != totalChunks {
		return fmt.Errorf("incomplete upload: received %d/%d chunks", receivedChunks, totalChunks)
	}

	// Assemble weights from chunks
	allWeights := make([]float64, 0, int(chunkState["totalParams"].(float64)))

	for i := 0; i < totalChunks; i++ {
		chunkKey := fmt.Sprintf("vpsa-weight-chunk-%d", i)
		chunkBytes, err := ctx.GetStub().GetPrivateData(globalModelWeightsCollection, chunkKey)
		if err != nil {
			return fmt.Errorf("failed to get chunk %d: %v", i, err)
		}
		if chunkBytes == nil {
			return fmt.Errorf("missing chunk %d", i)
		}

		var chunkWeights []float64
		if err := json.Unmarshal(chunkBytes, &chunkWeights); err != nil {
			return fmt.Errorf("failed to unmarshal chunk %d: %v", i, err)
		}

		allWeights = append(allWeights, chunkWeights...)

		// Clean up chunk
		if err := ctx.GetStub().DelPrivateData(globalModelWeightsCollection, chunkKey); err != nil {
			return fmt.Errorf("failed to delete chunk %d: %v", i, err)
		}
	}

	// Update global model
	gm, err := c.GetGlobalModel(ctx)
	if err != nil {
		return err
	}

	weightsHash, err := c.storeGlobalWeightsPrivate(ctx, allWeights)
	if err != nil {
		return err
	}

	gm.Weights = ""
	gm.WeightsHash = weightsHash
	gm.Version++
	timestamp, _ := getTxTimestamp(ctx)
	gm.Timestamp = timestamp
	gm.Status = "initialized"

	gmBytes, err := json.Marshal(gm)
	if err != nil {
		return fmt.Errorf("marshal global model failed: %v", err)
	}

	if err := ctx.GetStub().PutState("vpsa-global-model", gmBytes); err != nil {
		return fmt.Errorf("put global model failed: %v", err)
	}

	// Clean up chunk state
	if err := ctx.GetStub().DelState("vpsa-weights-chunked-upload"); err != nil {
		return fmt.Errorf("failed to cleanup chunk state: %v", err)
	}

	return nil
}

// ---------- Chunked Local Model Submission ----------

// SubmitLocalModelChunkedPart stores a single chunk of model weights into a private data collection.
// Uses transient data to keep weights private during submission.
// Transient key "chunkData" must contain JSON: {"modelID":"...", "clientID":"...", "collection":"collectionOrg1Private", "chunkIndex":0, "weights":[...]}
func (c *VPSAContract) SubmitLocalModelChunkedPart(ctx contractapi.TransactionContextInterface) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("failed to get transient: %v", err)
	}

	tBytes, ok := transientMap["chunkData"]
	if !ok {
		return fmt.Errorf("transient map must contain 'chunkData'")
	}

	// Decode base64 if needed
	decoded, err := base64.StdEncoding.DecodeString(string(tBytes))
	if err != nil {
		decoded = tBytes
	}

	var payload struct {
		ModelID    string    `json:"modelID"`
		ClientID   string    `json:"clientID"`
		Collection string    `json:"collection"`
		ChunkIndex int       `json:"chunkIndex"`
		Weights    []float64 `json:"weights"`
	}
	if err := json.Unmarshal(decoded, &payload); err != nil {
		return fmt.Errorf("invalid chunkData JSON: %v", err)
	}
	if payload.ModelID == "" || payload.ClientID == "" || payload.Collection == "" {
		return fmt.Errorf("modelID, clientID, and collection are required")
	}

	// Validate collection name
	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return fmt.Errorf("read config: %v", err)
	}
	found := false
	for _, cc := range cfg.Collections {
		if cc == payload.Collection {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("collection %s not recognized in config", payload.Collection)
	}

	// Store chunk in private collection with temp key
	chunkKey := fmt.Sprintf("%s::%s::chunk_%d", payload.ModelID, payload.ClientID, payload.ChunkIndex)
	chunkBytes, err := json.Marshal(payload.Weights)
	if err != nil {
		return fmt.Errorf("marshal chunk weights: %v", err)
	}

	if err := ctx.GetStub().PutPrivateData(payload.Collection, chunkKey, chunkBytes); err != nil {
		return fmt.Errorf("PutPrivateData chunk failed (%s): %v", payload.Collection, err)
	}

	return nil
}

// SubmitLocalModelChunkedCommit assembles chunks from private collections into the final ::part entries.
// Args: modelID, clientID, domain, numChunksOrg1 (string), numChunksOrg2 (string)
func (c *VPSAContract) SubmitLocalModelChunkedCommit(ctx contractapi.TransactionContextInterface,
	modelID string, clientID string, domain string, numChunksOrg1Str string, numChunksOrg2Str string) error {

	var numChunksOrg1, numChunksOrg2 int
	if _, err := fmt.Sscanf(numChunksOrg1Str, "%d", &numChunksOrg1); err != nil {
		return fmt.Errorf("invalid numChunksOrg1: %v", err)
	}
	if _, err := fmt.Sscanf(numChunksOrg2Str, "%d", &numChunksOrg2); err != nil {
		return fmt.Errorf("invalid numChunksOrg2: %v", err)
	}

	collections := []struct {
		name      string
		numChunks int
	}{
		{"collectionOrg1Private", numChunksOrg1},
		{"collectionOrg2Private", numChunksOrg2},
	}

	for _, coll := range collections {
		if coll.numChunks <= 0 {
			continue
		}

		// Read all chunks from this collection and assemble
		var assembled []float64
		for i := 0; i < coll.numChunks; i++ {
			chunkKey := fmt.Sprintf("%s::%s::chunk_%d", modelID, clientID, i)
			chunkBytes, err := ctx.GetStub().GetPrivateData(coll.name, chunkKey)
			if err != nil {
				return fmt.Errorf("GetPrivateData chunk failed (%s, chunk %d): %v", coll.name, i, err)
			}
			if chunkBytes == nil {
				return fmt.Errorf("missing chunk %d in %s for %s::%s", i, coll.name, modelID, clientID)
			}

			var chunkWeights []float64
			if err := json.Unmarshal(chunkBytes, &chunkWeights); err != nil {
				return fmt.Errorf("unmarshal chunk %d: %v", i, err)
			}
			assembled = append(assembled, chunkWeights...)

			// Clean up chunk
			if err := ctx.GetStub().DelPrivateData(coll.name, chunkKey); err != nil {
				return fmt.Errorf("cleanup chunk %d: %v", i, err)
			}
		}

		// Store assembled weights with the final key
		partKey := fmt.Sprintf("%s::%s::part", modelID, clientID)
		partBytes, err := json.Marshal(assembled)
		if err != nil {
			return fmt.Errorf("marshal assembled weights: %v", err)
		}

		if err := ctx.GetStub().PutPrivateData(coll.name, partKey, partBytes); err != nil {
			return fmt.Errorf("PutPrivateData assembled: %v", err)
		}
	}

	// Save metadata to public state
	meta := LocalModelMeta{
		ModelID:  modelID,
		ClientID: clientID,
		Round:    0,
		DocType:  "localModelMeta",
		Status:   "submitted",
	}
	metaKey := fmt.Sprintf("%s::%s::meta", modelID, clientID)
	metaB, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("meta marshal: %v", err)
	}
	if err := ctx.GetStub().PutState(metaKey, metaB); err != nil {
		return fmt.Errorf("put meta: %v", err)
	}

	// Update client info
	clientJSON, _ := ctx.GetStub().GetState(clientID)
	if clientJSON != nil {
		var client Client
		_ = json.Unmarshal(clientJSON, &client)
		ts, _ := getTxTimestamp(ctx)
		client.LastUpdate = ts
		client.IsActive = true
		if b, _ := json.Marshal(client); b != nil {
			_ = ctx.GetStub().PutState(clientID, b)
		}
	}

	// Register model ID in round list
	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return err
	}
	roundKey := fmt.Sprintf("round-%d-models", cfg.CurrentRound)
	roundModelsJSON, _ := ctx.GetStub().GetState(roundKey)
	var roundModels []string
	if roundModelsJSON != nil {
		_ = json.Unmarshal(roundModelsJSON, &roundModels)
	}
	found := false
	for _, m := range roundModels {
		if m == modelID {
			found = true
			break
		}
	}
	if !found {
		roundModels = append(roundModels, modelID)
		if rmj, err := json.Marshal(roundModels); err == nil {
			_ = ctx.GetStub().PutState(roundKey, rmj)
		}
	}

	return nil
}

// ---------- Transient submit and private data storage ----------

// SubmitLocalModelTransient expects the client to send a transient map containing key "localModelParts"
// The value must be JSON of the form:
//
//	{
//	  "modelID":"m1",
//	  "clientID":"clientA",
//	  "parts": {
//	     "collectionOrg1Private": [0.12, 0.34, ...],
//	     "collectionOrg2Private": [ ... ]
//	  }
//	}
//
// This method stores each part into its respective private data collection using key: "<modelID>::<clientID>::part"
func (c *VPSAContract) SubmitLocalModelTransient(ctx contractapi.TransactionContextInterface) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("failed to get transient: %v", err)
	}
	tBytes, ok := transientMap["localModelParts"]
	if !ok {
		return fmt.Errorf("transient map must contain 'localModelParts'")
	}

	// Decode base64 if needed (Flask sends base64-encoded data)
	decoded, err := base64.StdEncoding.DecodeString(string(tBytes))
	if err != nil {
		// If decode fails, assume it's already JSON (backward compatibility)
		decoded = tBytes
	}

	// parse top-level
	var payload struct {
		ModelID  string                     `json:"modelID"`
		ClientID string                     `json:"clientID"`
		Domain   string                     `json:"domain"`
		Parts    map[string][]float64       `json:"parts"`
		Meta     map[string]interface{}     `json:"meta,omitempty"`
		Labels   map[string]string          `json:"labels,omitempty"`
		Extra    map[string]json.RawMessage `json:"extra,omitempty"`
	}
	if err := json.Unmarshal(decoded, &payload); err != nil {
		return fmt.Errorf("invalid localModelParts json: %v", err)
	}
	if payload.ModelID == "" || payload.ClientID == "" {
		return fmt.Errorf("modelID and clientID are required")
	}

	// Save only metadata to public state (not the partitions themselves)
	// store LocalModelMeta under key modelID::clientID::meta
	meta := LocalModelMeta{
		ModelID:  payload.ModelID,
		ClientID: payload.ClientID,
		Round:    0,
		DocType:  "localModelMeta",
		Status:   "submitted",
	}
	metaKey := fmt.Sprintf("%s::%s::meta", payload.ModelID, payload.ClientID)
	if metaB, err := json.Marshal(meta); err == nil {
		if err := ctx.GetStub().PutState(metaKey, metaB); err != nil {
			return fmt.Errorf("put meta failed: %v", err)
		}
	} else {
		return fmt.Errorf("meta marshal failed: %v", err)
	}

	// store parts into private collections (collection names MUST match network collections_config.json)
	for collName, partVec := range payload.Parts {
		// ensure collection name appears in config
		cfg, err := c.GetAggregationConfig(ctx)
		if err != nil {
			return fmt.Errorf("read cfg: %v", err)
		}
		found := false
		for _, cc := range cfg.Collections {
			if cc == collName {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("collection %s not recognized in config", collName)
		}

		partBytes, err := json.Marshal(partVec)
		if err != nil {
			return fmt.Errorf("marshal partition failed: %v", err)
		}
		partKey := fmt.Sprintf("%s::%s::part", payload.ModelID, payload.ClientID)
		fmt.Printf("DEBUG: Storing to collection=%s, key=%s, dataLen=%d\n", collName, partKey, len(partVec))
		if err := ctx.GetStub().PutPrivateData(collName, partKey, partBytes); err != nil {
			return fmt.Errorf("PutPrivateData failed (%s): %v", collName, err)
		}
		fmt.Printf("DEBUG: Successfully stored partition in %s\n", collName)
	}

	// Update client last update + dataset info if provided
	if payload.ClientID != "" {
		clientJSON, _ := ctx.GetStub().GetState(payload.ClientID)
		if clientJSON != nil {
			var client Client
			_ = json.Unmarshal(clientJSON, &client)
			ts, _ := getTxTimestamp(ctx)
			client.LastUpdate = ts
			client.IsActive = true
			if b, _ := json.Marshal(client); b != nil {
				_ = ctx.GetStub().PutState(payload.ClientID, b)
			}
		}
	}

	// register model ID into the round list
	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return err
	}
	roundKey := fmt.Sprintf("round-%d-models", cfg.CurrentRound)
	roundModelsJSON, _ := ctx.GetStub().GetState(roundKey)
	var roundModels []string
	if roundModelsJSON != nil {
		_ = json.Unmarshal(roundModelsJSON, &roundModels)
	}
	// Add modelID if not present
	found := false
	for _, m := range roundModels {
		if m == payload.ModelID {
			found = true
			break
		}
	}
	if !found {
		roundModels = append(roundModels, payload.ModelID)
		if rmj, err := json.Marshal(roundModels); err == nil {
			if err := ctx.GetStub().PutState(roundKey, rmj); err != nil {
				return err
			}
		}
	}

	return nil
}

// NOTE: fetchPartitionsForModel was DELETED
// VPSA paper requirement: No entity ever observes a complete client update vector.
// Reconstructing full vectors violates the VPSA threat model and n-1 collusion guarantee.

// AggregateSliceVPSA aggregates updates for a SINGLE vertical slice (collection).
// This is the core VPSA operation - each slice is aggregated independently.
// No peer/server ever sees the complete client vector.
//
// Paper reference: "VPSA aggregation happens per slice"
// Security guarantee: Even with n-1 colluding servers, no full vector is revealed.
func (c *VPSAContract) AggregateSliceVPSA(
	ctx contractapi.TransactionContextInterface,
	collectionName string,
	modelClientIDs []string,
	globalSlice []float64,
	beta int,
) ([]float64, error) {
	if len(modelClientIDs) == 0 {
		return nil, fmt.Errorf("no model-client IDs provided")
	}

	// Collect client slices for THIS collection only
	type clientSlice struct {
		clientID string
		slice    []float64
	}
	var slices []clientSlice

	for _, mc := range modelClientIDs {
		parts := strings.Split(mc, "::")
		if len(parts) != 2 {
			return nil, fmt.Errorf("modelClientIDs must be modelID::clientID; got %s", mc)
		}
		modelID := parts[0]
		clientID := parts[1]

		// Read ONLY from this collection - never touch other collections
		partKey := fmt.Sprintf("%s::%s::part", modelID, clientID)
		partBytes, err := ctx.GetStub().GetPrivateData(collectionName, partKey)
		if err != nil {
			return nil, fmt.Errorf("GetPrivateData failed for %s: %v", collectionName, err)
		}
		if partBytes == nil {
			return nil, fmt.Errorf("partition missing in %s for %s", collectionName, partKey)
		}

		sliceVec, err := floatSliceFromJSONBytes(partBytes)
		if err != nil {
			return nil, fmt.Errorf("invalid partition JSON: %v", err)
		}

		slices = append(slices, clientSlice{clientID: clientID, slice: sliceVec})
	}

	numClients := len(slices)
	if numClients == 0 {
		return nil, fmt.Errorf("no client slices found in %s", collectionName)
	}

	sliceLen := len(slices[0].slice)
	for _, s := range slices {
		if len(s.slice) != sliceLen {
			return nil, fmt.Errorf("slice length mismatch in %s", collectionName)
		}
	}

	// PAPER-EXACT: Compute deltas Δw_i = w_global^(k) - v_i^(k)
	// Paper Algorithm 1: model update m_i is the difference from global model
	deltas := make([][]float64, numClients)
	for i, s := range slices {
		delta := make([]float64, sliceLen)
		if globalSlice != nil && len(globalSlice) == sliceLen {
			for k := 0; k < sliceLen; k++ {
				delta[k] = globalSlice[k] - s.slice[k]
			}
		} else {
			// No global reference: treat client values as deltas directly
			copy(delta, s.slice)
		}
		deltas[i] = delta
	}

	// PAPER-EXACT: cos_weights[i] = cos(Δw_i, w_global^(k))
	// Raw cosine similarity in [-1, 1] range — NO normalization to [0,1]
	cosWeights := make([]float64, numClients)
	for i := range deltas {
		if globalSlice != nil && len(globalSlice) == sliceLen {
			cs, err := cosineSimilarity(deltas[i], globalSlice)
			if err != nil {
				cs = 0
			}
			cosWeights[i] = cs // Paper-exact: raw cosine [-1, 1]
		} else {
			cosWeights[i] = 1.0
		}
	}

	// PAPER-EXACT VPSA: Algorithm 1, lines 7-16
	// For each coordinate k, compute:
	// 1. Euclidean weights (binary 0/1 based on β-trimming)
	// 2. Cosine weights (raw cosine, computed above)
	// 3. Final weight = euclidean × cosine
	// 4. Aggregated delta = sum(weight × delta) / sum(weight)

	aggregatedSlice := make([]float64, sliceLen)

	for k := 0; k < sliceLen; k++ {
		// Collect (delta_value, clientIndex) pairs for coordinate k
		type pair struct {
			val   float64
			index int
		}
		arr := make([]pair, numClients)
		for i := 0; i < numClients; i++ {
			arr[i] = pair{val: deltas[i][k], index: i}
		}

		// Sort deltas by value for trimming (Algorithm 1, line 8)
		sort.Slice(arr, func(i, j int) bool { return arr[i].val < arr[j].val })

		// PAPER-EXACT: Compute Euclidean weights (binary trimming)
		// Algorithm 1, lines 9-13: w_euc = 0 for trimmed, 1 otherwise
		euclideanWeights := make([]float64, numClients)
		trimLow, trimHigh := 0, 0
		if beta > 0 && beta*2 < numClients {
			trimLow = beta
			trimHigh = beta
		}

		for idx := 0; idx < numClients; idx++ {
			if idx < trimLow || idx >= (numClients-trimHigh) {
				euclideanWeights[arr[idx].index] = 0.0 // TRIMMED
			} else {
				euclideanWeights[arr[idx].index] = 1.0 // KEPT
			}
		}

		// PAPER-EXACT: Combine Euclidean × Cosine weights (Algorithm 1, line 14)
		var numSum, denSum float64
		for i := 0; i < numClients; i++ {
			finalWeight := euclideanWeights[i] * cosWeights[i]
			numSum += finalWeight * deltas[i][k]
			denSum += finalWeight
		}

		// Algorithm 1, line 15: aggregated delta coordinate
		aggregatedSlice[k] = safeDiv(numSum, denSum)
	}

	// Return aggregated DELTA slice (paper-exact Δw_agg for this vertical partition)
	return aggregatedSlice, nil
}

// AggregateModelsVPSA implements paper-compliant VPSA aggregation.
//
// CRITICAL: This function NEVER reconstructs full client vectors.
// Instead, it aggregates each vertical slice independently, then
// concatenates the aggregated deltas to update the global model.
//
// Paper compliance:
// - Each collection (vertical slice) is aggregated separately
// - Cosine similarity is computed per-slice only
// - No peer ever sees a complete client update
// - Global model update: w_{t+1} = w_t - concat(Δw^(1), Δw^(2), ...)
func (c *VPSAContract) AggregateModelsVPSA(ctx contractapi.TransactionContextInterface, modelClientIDs []string, beta int) error {
	if len(modelClientIDs) == 0 {
		return fmt.Errorf("no model-client IDs provided")
	}

	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return err
	}
	gm, err := c.GetGlobalModel(ctx)
	if err != nil {
		return err
	}

	globalWeights, err := c.loadGlobalWeightsPrivate(ctx, gm)
	if err != nil {
		return err
	}

	numCollections := len(cfg.Collections)
	if numCollections == 0 {
		return fmt.Errorf("no collections configured")
	}

	// PAPER-EXACT: Determine total weight vector size
	// Compute from global model or by summing all collection slice sizes
	totalSize := len(globalWeights)
	if totalSize == 0 {
		// Infer total size from client data across all collections
		sampleModelClient := strings.Split(modelClientIDs[0], "::")
		if len(sampleModelClient) != 2 {
			return fmt.Errorf("invalid modelClientID format: %s", modelClientIDs[0])
		}
		for _, collName := range cfg.Collections {
			sampleKey := fmt.Sprintf("%s::%s::part", sampleModelClient[0], sampleModelClient[1])
			sampleBytes, err := ctx.GetStub().GetPrivateData(collName, sampleKey)
			if err != nil || sampleBytes == nil {
				continue
			}
			sampleSlice, _ := floatSliceFromJSONBytes(sampleBytes)
			totalSize += len(sampleSlice)
		}
	}
	if totalSize == 0 {
		return fmt.Errorf("cannot determine total model size from global model or client data")
	}

	// PAPER-EXACT: Modulo-based vertical partitioning
	// Paper rule: position j → collection ((totalSize-1-j) %% numCollections)
	partitionIndices := paperPartitionIndices(totalSize, numCollections)

	// Aggregate each vertical slice independently (paper Algorithm 1)
	// AggregateSliceVPSA now returns aggregated DELTAS per slice
	aggregatedDeltas := make([]float64, totalSize)

	for collIdx, collName := range cfg.Collections {
		indices := partitionIndices[collIdx]

		// Extract the global weights for THIS partition's positions
		var globalSlice []float64
		if globalWeights != nil && len(globalWeights) == totalSize {
			globalSlice = extractByIndices(globalWeights, indices)
		}

		// Aggregate THIS slice only — never see other collections
		aggDeltaSlice, err := c.AggregateSliceVPSA(ctx, collName, modelClientIDs, globalSlice, beta)
		if err != nil {
			return fmt.Errorf("AggregateSliceVPSA failed for %s: %v", collName, err)
		}

		// Scatter aggregated deltas back to their original positions
		scatterByIndices(aggregatedDeltas, indices, aggDeltaSlice)
	}

	// PAPER-EXACT: w_{t+1} = w_t − Δw_agg (Algorithm 1, final line)
	// Delta-based global model update, NOT direct replacement
	newGlobal := make([]float64, totalSize)
	if globalWeights != nil && len(globalWeights) == totalSize {
		for i := 0; i < totalSize; i++ {
			newGlobal[i] = globalWeights[i] - aggregatedDeltas[i]
		}
	} else {
		// First round: no previous global weights, use negated deltas as initial
		for i := 0; i < totalSize; i++ {
			newGlobal[i] = -aggregatedDeltas[i]
		}
	}

	weightsHash, err := c.storeGlobalWeightsPrivate(ctx, newGlobal)
	if err != nil {
		return err
	}
	gm.Weights = ""
	gm.WeightsHash = weightsHash
	gm.Version++
	gm.Round = cfg.CurrentRound
	gm.Timestamp, _ = getTxTimestamp(ctx)
	gm.Status = "updated"

	gmBytes, _ := json.Marshal(gm)
	if err := ctx.GetStub().PutState("vpsa-global-model", gmBytes); err != nil {
		return fmt.Errorf("put global model failed: %v", err)
	}

	// Mark local models as aggregated
	for _, mc := range modelClientIDs {
		parts := strings.Split(mc, "::")
		if len(parts) != 2 {
			continue
		}
		metaKey := fmt.Sprintf("%s::%s::meta", parts[0], parts[1])
		metaJSON, _ := ctx.GetStub().GetState(metaKey)
		if metaJSON != nil {
			var meta LocalModelMeta
			_ = json.Unmarshal(metaJSON, &meta)
			meta.Status = "aggregated"
			if b, _ := json.Marshal(meta); b != nil {
				_ = ctx.GetStub().PutState(metaKey, b)
			}
		}
	}

	// Store training metrics
	metrics := TrainingMetrics{
		MetricID:        fmt.Sprintf("metrics-round-%d", cfg.CurrentRound),
		Round:           cfg.CurrentRound,
		GlobalAccuracy:  0.0,
		GlobalLoss:      0.0,
		AlignmentScore:  0.0,
		NumParticipants: len(modelClientIDs),
		Timestamp:       gm.Timestamp,
	}
	metricsBytes, _ := json.Marshal(metrics)
	_ = ctx.GetStub().PutState(metrics.MetricID, metricsBytes)

	// Advance round
	cfg.CurrentRound++
	cfg.LastUpdated = gm.Timestamp
	cfgBytes, _ := json.Marshal(cfg)
	_ = ctx.GetStub().PutState("vpsa-config", cfgBytes)

	return nil
}

// ---------- Secure prediction (simple additive secret shares) ----------

// Split query vector into N additive shares and put into private collections
// transient must contain "queryID" and "queryVec" (JSON []float64)
// This function splits into equal-length share arrays and stores share i in cfg.Collections[i]
func (c *VPSAContract) SecurePredictSetup(ctx contractapi.TransactionContextInterface, queryID string) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("get transient failed: %v", err)
	}
	qBytes, ok := transientMap["queryVec"]
	if !ok {
		return fmt.Errorf("transient must contain queryVec")
	}
	var queryVec []float64
	if err := json.Unmarshal(qBytes, &queryVec); err != nil {
		return fmt.Errorf("invalid queryVec json: %v", err)
	}

	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return err
	}
	n := len(cfg.Collections)
	if n == 0 {
		return fmt.Errorf("no collections configured")
	}

	// Create additive shares such that sum(shares) = queryVec
	// Simple equal split for demonstration
	shares := make([][]float64, n)
	for i := 0; i < n; i++ {
		shares[i] = make([]float64, len(queryVec))
	}

	// Split equally across n shares
	for i := 0; i < len(queryVec); i++ {
		val := queryVec[i]
		portion := val / float64(n)
		for s := 0; s < n-1; s++ {
			shares[s][i] = portion
		}
		shares[n-1][i] = val - portion*float64(n-1)
	}

	// Store shares into private collections WITHOUT index suffix
	// Key format: "query::<queryID>::share"
	for i, coll := range cfg.Collections {
		key := fmt.Sprintf("query::%s::share", queryID)
		b, _ := json.Marshal(shares[i])
		if err := ctx.GetStub().PutPrivateData(coll, key, b); err != nil {
			return fmt.Errorf("put private share failed for %s: %v", coll, err)
		}
	}

	return nil
}

// ReconstructResult expects transient containing partial logits stored in collections by peers and reconstructs
// For demonstration, we'll gather private data keys "query::<queryID>::logits::<i>"
func (c *VPSAContract) SecurePredictReconstruct(ctx contractapi.TransactionContextInterface, queryID string) ([]float64, error) {
	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return nil, err
	}
	// collect logits shares and sum them
	var result []float64
	for i, coll := range cfg.Collections {
		key := fmt.Sprintf("query::%s::logits::%d", queryID, i)
		b, err := ctx.GetStub().GetPrivateData(coll, key)
		if err != nil {
			return nil, fmt.Errorf("get logits share failed: %v", err)
		}
		if b == nil {
			return nil, fmt.Errorf("logits share missing in coll %s key %s", coll, key)
		}
		var share []float64
		if err := json.Unmarshal(b, &share); err != nil {
			return nil, fmt.Errorf("invalid logits share: %v", err)
		}
		if result == nil {
			result = make([]float64, len(share))
		}
		if len(share) != len(result) {
			return nil, fmt.Errorf("logits share length mismatch")
		}
		for j := range share {
			result[j] += share[j]
		}
		_ = i // just to avoid unused, we use i above
	}
	return result, nil
}

// ...existing code...

// ComputePartialPrediction computes a partial prediction using the query share in this org's collection
// and the global model weights. Stores the partial logits in private collection.
// transient must contain "queryID"
func (s *VPSAContract) ComputePartialPrediction(ctx contractapi.TransactionContextInterface) error {
	// Get query ID from transient
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("error getting transient: %v", err)
	}

	queryIDBytes, ok := transientMap["queryID"]
	if !ok {
		return errors.New("queryID not found in transient map")
	}
	queryID := strings.TrimSpace(string(queryIDBytes))

	// Get aggregation config
	cfgBytes, err := ctx.GetStub().GetState("vpsa-config")
	if err != nil {
		return fmt.Errorf("failed to read config: %v", err)
	}
	var cfg AggregationConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return err
	}

	// Get global model
	gmBytes, err := ctx.GetStub().GetState("vpsa-global-model")
	if err != nil {
		return fmt.Errorf("failed to read global model: %v", err)
	}
	var gm GlobalModel
	if err := json.Unmarshal(gmBytes, &gm); err != nil {
		return err
	}

	weights, err := s.loadGlobalWeightsPrivate(ctx, &gm)
	if err != nil {
		return err
	}

	if len(weights) == 0 {
		return errors.New("global model has no weights")
	}

	// Get client MSP ID to determine which collection to use
	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get client MSP ID: %v", err)
	}

	// Determine collection based on MSP
	var myCollection string
	if clientMSPID == "Org1MSP" {
		myCollection = "collectionOrg1Private"
	} else if clientMSPID == "Org2MSP" {
		myCollection = "collectionOrg2Private"
	} else {
		return fmt.Errorf("unknown MSP ID: %s", clientMSPID)
	}

	// Read query share from private collection (key matches what SecurePredictSetup stored)
	shareKey := fmt.Sprintf("query::%s::share", queryID)
	shareBytes, err := ctx.GetStub().GetPrivateData(myCollection, shareKey)
	if err != nil {
		return fmt.Errorf("failed to read query share from %s: %v", myCollection, err)
	}
	if shareBytes == nil {
		return fmt.Errorf("query share not found in %s for queryID: %s", myCollection, queryID)
	}

	var queryShare []float64
	if err := json.Unmarshal(shareBytes, &queryShare); err != nil {
		return fmt.Errorf("failed to unmarshal query share: %v", err)
	}

	// Ensure dimensions match
	if len(queryShare) != len(weights) {
		return fmt.Errorf("dimension mismatch: query share has %d elements, model has %d weights", len(queryShare), len(weights))
	}

	// Compute partial prediction: dot product
	partialLogit := 0.0
	for i := 0; i < len(queryShare); i++ {
		partialLogit += queryShare[i] * weights[i]
	}

	// Store partial logit in private collection
	// PAPER-FIX: Use consistent key format with collection name
	logitKey := fmt.Sprintf("query::%s::logit::%s", queryID, myCollection)
	logitBytes, err := json.Marshal(partialLogit)
	if err != nil {
		return err
	}

	if err := ctx.GetStub().PutPrivateData(myCollection, logitKey, logitBytes); err != nil {
		return fmt.Errorf("failed to store partial logit: %v", err)
	}

	return nil
}

// ReconstructPrediction - Same as before
func (s *VPSAContract) ReconstructPrediction(ctx contractapi.TransactionContextInterface) error {
	// Get query ID from transient
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("error getting transient: %v", err)
	}

	queryIDBytes, ok := transientMap["queryID"]
	if !ok {
		return errors.New("queryID not found in transient map")
	}
	queryID := strings.TrimSpace(string(queryIDBytes))

	// Get aggregation config
	cfgBytes, err := ctx.GetStub().GetState("vpsa-config")
	if err != nil {
		return fmt.Errorf("failed to read config: %v", err)
	}
	var cfg AggregationConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return err
	}

	// Gather partial logits from all collections
	// PAPER-FIX: Use consistent key format matching ComputePartialPrediction
	totalLogit := 0.0

	for _, collName := range cfg.Collections {
		logitKey := fmt.Sprintf("query::%s::logit::%s", queryID, collName)
		logitBytes, err := ctx.GetStub().GetPrivateData(collName, logitKey)
		if err != nil {
			return fmt.Errorf("failed to read partial logit from %s: %v", collName, err)
		}
		if logitBytes == nil {
			return fmt.Errorf("partial logit not found in collection %s for queryID: %s", collName, queryID)
		}

		var partialLogit float64
		if err := json.Unmarshal(logitBytes, &partialLogit); err != nil {
			return err
		}

		totalLogit += partialLogit
	}

	// Apply sigmoid activation for binary classification
	finalPrediction := 1.0 / (1.0 + math.Exp(-totalLogit))

	// Store result in public state
	result := struct {
		QueryID    string  `json:"queryID"`
		Logit      float64 `json:"logit"`
		Prediction float64 `json:"prediction"`
		Timestamp  string  `json:"timestamp"`
	}{
		QueryID:    queryID,
		Logit:      totalLogit,
		Prediction: finalPrediction,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}

	resultBytes, err := json.Marshal(result)
	if err != nil {
		return err
	}

	resultKey := fmt.Sprintf("prediction::%s", queryID)
	if err := ctx.GetStub().PutState(resultKey, resultBytes); err != nil {
		return fmt.Errorf("failed to store prediction result: %v", err)
	}

	return nil
}

// GetPrediction retrieves a stored prediction result
func (s *VPSAContract) GetPrediction(ctx contractapi.TransactionContextInterface, queryID string) (string, error) {
	resultKey := fmt.Sprintf("prediction::%s", queryID)
	resultBytes, err := ctx.GetStub().GetState(resultKey)
	if err != nil {
		return "", fmt.Errorf("failed to read prediction: %v", err)
	}
	if resultBytes == nil {
		return "", fmt.Errorf("prediction not found for queryID: %s", queryID)
	}

	return string(resultBytes), nil
}

// ==================== ZERO-EXPOSURE SECURE INFERENCE ====================
// Novelty: The full query vector NEVER enters any chaincode function or peer.
// The client performs local pre-partitioning into additive secret shares and
// sends each share directly to the corresponding org's private collection.
// No endorsing peer ever sees the raw input. This eliminates the trusted
// endorsing peer assumption present in standard secure inference.
// =========================================================================

// ZEStoreQueryShare stores a single pre-computed query share in the calling
// org's private collection. The full query vector never touches any peer.
//
// Transient data required:
//   - "shareData":  JSON-encoded []float64 (the share for THIS org)
//   - "queryID":    string identifier for the query
//   - "commitment": hex-encoded SHA-256 hash commitment of this share
//   - "shareIndex": "0" or "1" indicating which share this is
//   - "totalShares": total number of shares (e.g. "2")
func (c *VPSAContract) ZEStoreQueryShare(ctx contractapi.TransactionContextInterface) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("failed to get transient data: %v", err)
	}

	// --- Extract transient fields ---
	shareDataBytes, ok := transientMap["shareData"]
	if !ok {
		return errors.New("transient must contain 'shareData'")
	}
	queryIDBytes, ok := transientMap["queryID"]
	if !ok {
		return errors.New("transient must contain 'queryID'")
	}
	commitmentBytes, ok := transientMap["commitment"]
	if !ok {
		return errors.New("transient must contain 'commitment'")
	}
	shareIndexBytes, ok := transientMap["shareIndex"]
	if !ok {
		return errors.New("transient must contain 'shareIndex'")
	}

	// Decode values (handle possible base64 wrapping from Fabric CLI)
	shareDataRaw := shareDataBytes
	if decoded, err2 := base64.StdEncoding.DecodeString(string(shareDataBytes)); err2 == nil {
		shareDataRaw = decoded
	}
	queryID := strings.TrimSpace(string(queryIDBytes))
	if decoded, err2 := base64.StdEncoding.DecodeString(queryID); err2 == nil {
		queryID = strings.TrimSpace(string(decoded))
	}
	commitmentHex := strings.TrimSpace(string(commitmentBytes))
	if decoded, err2 := base64.StdEncoding.DecodeString(commitmentHex); err2 == nil {
		commitmentHex = strings.TrimSpace(string(decoded))
	}
	shareIndex := strings.TrimSpace(string(shareIndexBytes))
	if decoded, err2 := base64.StdEncoding.DecodeString(shareIndex); err2 == nil {
		shareIndex = strings.TrimSpace(string(decoded))
	}

	// Validate share data is valid JSON float array
	var shareVec []float64
	if err := json.Unmarshal(shareDataRaw, &shareVec); err != nil {
		return fmt.Errorf("invalid shareData JSON: %v", err)
	}
	if len(shareVec) == 0 {
		return errors.New("shareData must not be empty")
	}

	// Verify the commitment: SHA-256(shareData raw bytes) must match
	hash := sha256.Sum256(shareDataRaw)
	computedHex := hex.EncodeToString(hash[:])
	if computedHex != commitmentHex {
		return fmt.Errorf("commitment verification failed: expected %s, got %s", commitmentHex, computedHex)
	}

	// Determine which private collection belongs to the caller
	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get client MSP ID: %v", err)
	}
	var myCollection string
	switch clientMSPID {
	case "Org1MSP":
		myCollection = "collectionOrg1Private"
	case "Org2MSP":
		myCollection = "collectionOrg2Private"
	default:
		return fmt.Errorf("unknown MSP ID: %s", clientMSPID)
	}

	// Store the share in the org's private collection
	// Key format: "ze-query::<queryID>::share"
	shareKey := fmt.Sprintf("ze-query::%s::share", queryID)
	if err := ctx.GetStub().PutPrivateData(myCollection, shareKey, shareDataRaw); err != nil {
		return fmt.Errorf("failed to store share in %s: %v", myCollection, err)
	}

	// Store the commitment on the PUBLIC ledger so all parties can audit
	// Key: "ze-commitment::<queryID>::<shareIndex>"
	commitKey := fmt.Sprintf("ze-commitment::%s::%s", queryID, shareIndex)
	commitRecord := struct {
		QueryID    string `json:"queryID"`
		ShareIndex string `json:"shareIndex"`
		Commitment string `json:"commitment"`
		Dimension  int    `json:"dimension"`
		OrgMSP     string `json:"orgMSP"`
		Timestamp  string `json:"timestamp"`
	}{
		QueryID:    queryID,
		ShareIndex: shareIndex,
		Commitment: computedHex,
		Dimension:  len(shareVec),
		OrgMSP:     clientMSPID,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}
	commitBytes, _ := json.Marshal(commitRecord)
	if err := ctx.GetStub().PutState(commitKey, commitBytes); err != nil {
		return fmt.Errorf("failed to store commitment: %v", err)
	}

	// Track that this org has submitted its share
	statusKey := fmt.Sprintf("ze-status::%s", queryID)
	statusBytes, _ := ctx.GetStub().GetState(statusKey)
	var status map[string]bool
	if statusBytes != nil {
		json.Unmarshal(statusBytes, &status)
	} else {
		status = make(map[string]bool)
	}
	status[clientMSPID] = true
	newStatusBytes, _ := json.Marshal(status)
	if err := ctx.GetStub().PutState(statusKey, newStatusBytes); err != nil {
		return fmt.Errorf("failed to update share status: %v", err)
	}

	return nil
}

// ZEGetShareStatus returns which orgs have submitted their shares for a query
func (c *VPSAContract) ZEGetShareStatus(ctx contractapi.TransactionContextInterface, queryID string) (string, error) {
	statusKey := fmt.Sprintf("ze-status::%s", queryID)
	statusBytes, err := ctx.GetStub().GetState(statusKey)
	if err != nil {
		return "", fmt.Errorf("failed to read share status: %v", err)
	}
	if statusBytes == nil {
		return "{}", nil
	}
	return string(statusBytes), nil
}

// ZEComputePartialPrediction computes a partial dot-product prediction using
// the caller's private query share and the global model weights.
// This is similar to ComputePartialPrediction but reads from ze-query keys.
//
// Transient: {"queryID": "..."}
func (c *VPSAContract) ZEComputePartialPrediction(ctx contractapi.TransactionContextInterface) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("error getting transient: %v", err)
	}

	queryIDBytes, ok := transientMap["queryID"]
	if !ok {
		return errors.New("queryID not found in transient map")
	}
	queryID := strings.TrimSpace(string(queryIDBytes))
	if decoded, err2 := base64.StdEncoding.DecodeString(queryID); err2 == nil {
		queryID = strings.TrimSpace(string(decoded))
	}

	// Get global model
	gmBytes, err := ctx.GetStub().GetState("vpsa-global-model")
	if err != nil {
		return fmt.Errorf("failed to read global model: %v", err)
	}
	var gm GlobalModel
	if err := json.Unmarshal(gmBytes, &gm); err != nil {
		return err
	}

	weights, err := c.loadGlobalWeightsPrivate(ctx, &gm)
	if err != nil {
		return err
	}
	if len(weights) == 0 {
		return errors.New("global model has no weights")
	}

	// Determine caller's collection
	clientMSPID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get client MSP ID: %v", err)
	}
	var myCollection string
	switch clientMSPID {
	case "Org1MSP":
		myCollection = "collectionOrg1Private"
	case "Org2MSP":
		myCollection = "collectionOrg2Private"
	default:
		return fmt.Errorf("unknown MSP ID: %s", clientMSPID)
	}

	// Read this org's query share from ZE private data
	shareKey := fmt.Sprintf("ze-query::%s::share", queryID)
	shareBytes, err := ctx.GetStub().GetPrivateData(myCollection, shareKey)
	if err != nil {
		return fmt.Errorf("failed to read ZE query share from %s: %v", myCollection, err)
	}
	if shareBytes == nil {
		return fmt.Errorf("ZE query share not found in %s for queryID: %s", myCollection, queryID)
	}

	var queryShare []float64
	if err := json.Unmarshal(shareBytes, &queryShare); err != nil {
		return fmt.Errorf("failed to unmarshal ZE query share: %v", err)
	}

	// Dimension check
	if len(queryShare) != len(weights) {
		return fmt.Errorf("dimension mismatch: ZE share has %d elements, model has %d weights",
			len(queryShare), len(weights))
	}

	// Compute partial logit: dot(share, weights)
	partialLogit := 0.0
	for i := 0; i < len(queryShare); i++ {
		partialLogit += queryShare[i] * weights[i]
	}

	// Store partial logit in private collection
	logitKey := fmt.Sprintf("ze-query::%s::logit::%s", queryID, myCollection)
	logitBytes, err := json.Marshal(partialLogit)
	if err != nil {
		return err
	}
	if err := ctx.GetStub().PutPrivateData(myCollection, logitKey, logitBytes); err != nil {
		return fmt.Errorf("failed to store ZE partial logit: %v", err)
	}

	return nil
}

// ZEReconstructPrediction gathers partial logits from all orgs and produces
// the final prediction. Identical math to ReconstructPrediction but reads
// from ze-query keys. Also verifies commitment integrity.
//
// Transient: {"queryID": "..."}
func (c *VPSAContract) ZEReconstructPrediction(ctx contractapi.TransactionContextInterface) error {
	transientMap, err := ctx.GetStub().GetTransient()
	if err != nil {
		return fmt.Errorf("error getting transient: %v", err)
	}

	queryIDBytes, ok := transientMap["queryID"]
	if !ok {
		return errors.New("queryID not found in transient map")
	}
	queryID := strings.TrimSpace(string(queryIDBytes))
	if decoded, err2 := base64.StdEncoding.DecodeString(queryID); err2 == nil {
		queryID = strings.TrimSpace(string(decoded))
	}

	// Verify all shares were submitted
	statusKey := fmt.Sprintf("ze-status::%s", queryID)
	statusBytes, err := ctx.GetStub().GetState(statusKey)
	if err != nil {
		return fmt.Errorf("failed to read share status: %v", err)
	}
	var status map[string]bool
	if statusBytes != nil {
		json.Unmarshal(statusBytes, &status)
	}
	if !status["Org1MSP"] || !status["Org2MSP"] {
		return fmt.Errorf("not all shares submitted: org1=%v, org2=%v", status["Org1MSP"], status["Org2MSP"])
	}

	// Verify commitments exist on public ledger for auditability
	for i := 0; i < 2; i++ {
		commitKey := fmt.Sprintf("ze-commitment::%s::%d", queryID, i)
		cBytes, err := ctx.GetStub().GetState(commitKey)
		if err != nil || cBytes == nil {
			return fmt.Errorf("commitment missing for share %d of query %s", i, queryID)
		}
	}

	// Get aggregation config for collection names
	cfgBytes, err := ctx.GetStub().GetState("vpsa-config")
	if err != nil {
		return fmt.Errorf("failed to read config: %v", err)
	}
	var cfg AggregationConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return err
	}

	// Sum partial logits from all collections
	totalLogit := 0.0
	for _, collName := range cfg.Collections {
		logitKey := fmt.Sprintf("ze-query::%s::logit::%s", queryID, collName)
		logitBytes, err := ctx.GetStub().GetPrivateData(collName, logitKey)
		if err != nil {
			return fmt.Errorf("failed to read ZE partial logit from %s: %v", collName, err)
		}
		if logitBytes == nil {
			return fmt.Errorf("ZE partial logit not found in %s for query %s", collName, queryID)
		}

		var partialLogit float64
		if err := json.Unmarshal(logitBytes, &partialLogit); err != nil {
			return err
		}
		totalLogit += partialLogit
	}

	// Sigmoid activation
	finalPrediction := 1.0 / (1.0 + math.Exp(-totalLogit))

	// Store result on public ledger
	result := struct {
		QueryID      string  `json:"queryID"`
		Logit        float64 `json:"logit"`
		Prediction   float64 `json:"prediction"`
		ZeroExposure bool    `json:"zeroExposure"`
		Timestamp    string  `json:"timestamp"`
	}{
		QueryID:      queryID,
		Logit:        totalLogit,
		Prediction:   finalPrediction,
		ZeroExposure: true,
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
	}

	resultBytes, err := json.Marshal(result)
	if err != nil {
		return err
	}

	resultKey := fmt.Sprintf("prediction::%s", queryID)
	if err := ctx.GetStub().PutState(resultKey, resultBytes); err != nil {
		return fmt.Errorf("failed to store ZE prediction result: %v", err)
	}

	return nil
}

// ==========================================
// NOVELTY: Adaptive VPSA — Anomaly-Aware Aggregation
// Adapted to the VERTICAL SECURE setting.
//
// KEY SECURITY PROPERTY:
//   No full client vector is EVER reconstructed.
//   Anomaly scores are computed from per-slice PARTIAL SUMMARIES
//   (scalar values: dist², dot product, norm²) that do not reveal
//   individual model coordinates.
//
// Mathematical decomposition:
//   ||G_i - M||₂ = sqrt( Σ_p ||G_i^(p) - M^(p)||² )
//   cos(G_i, M)  = ( Σ_p G_i^(p)·M^(p) ) / ( ||G_i|| × ||M|| )
//   where ||G_i|| = sqrt( Σ_p ||G_i^(p)||² ), same for ||M||
//
// Algorithm:
//   For each slice/collection p:
//     1a. Compute coordinate-wise median M^(p) within this slice
//     1b. Compute partial scalars per client: distSq_p, dot_p, normGSq_p, normMSq_p
//   Then globally (from scalar partials only):
//     2. Assemble anomaly scores: Score_i = α*D_i + (1-α)*(1-C_i)
//     3. IQR threshold: T = Q3 + 1.5 * IQR
//     4. Filter: keep G_i where Score_i ≤ T
//   For each slice/collection p:
//     5. AggregateSliceVPSA on filtered client set only
// ==========================================

// AnomalyReport is stored on-chain for each adaptive aggregation round
type AnomalyReport struct {
	ReportID          string             `json:"reportID"`
	Round             int                `json:"round"`
	Alpha             float64            `json:"alpha"`
	Threshold         float64            `json:"threshold"`
	Q1                float64            `json:"q1"`
	Q3                float64            `json:"q3"`
	IQR               float64            `json:"iqr"`
	TotalClients      int                `json:"totalClients"`
	FilteredClients   int                `json:"filteredClients"`
	RejectedClients   []string           `json:"rejectedClients"`
	ClientScores      map[string]float64 `json:"clientScores"`
	EffectiveScores   map[string]float64 `json:"effectiveScores"`   // Trust-adjusted anomaly scores
	ClientTrustScores map[string]float64 `json:"clientTrustScores"` // Trust scores AFTER this round
	Timestamp         string             `json:"timestamp"`
	DocType           string             `json:"docType"`
}

// AdaptiveAggregateModelsVPSA implements Adaptive VPSA in the vertical secure setting.
//
// CRITICAL SECURITY PROPERTY:
//
//	This function NEVER reconstructs full client vectors.
//	Anomaly detection operates on per-slice partial scalar summaries,
//	which are then combined to produce global anomaly scores.
//	This preserves VPSA's vertical partitioning privacy guarantee:
//	even with n-1 colluding servers, no complete client update is revealed.
//
// Algorithm (Vertical-Secure Adaptive VPSA):
//
//	Phase A — Per-slice partial computation (private data access):
//	  For each collection p:
//	    Step 1a: Compute coordinate-wise median M^(p) for this slice
//	    Step 1b: For each client i, compute partial scalars:
//	      distSq_p(i) = Σ_{k∈S_p} (G_i[k] - M[k])²
//	      dot_p(i)    = Σ_{k∈S_p} G_i[k] × M[k]
//	      normGSq_p(i)= Σ_{k∈S_p} G_i[k]²
//	      normMSq_p   = Σ_{k∈S_p} M[k]²
//
//	Phase B — Global score assembly (from scalar partials only):
//	  Step 2: D_i = sqrt(Σ_p distSq_p(i))
//	          C_i = (Σ_p dot_p(i)) / (sqrt(Σ_p normGSq_p(i)) × sqrt(Σ_p normMSq_p))
//	          Score_i = α × D_i + (1-α) × (1 - C_i)
//	  Step 3: T = Q3 + 1.5 × IQR
//	  Step 4: Filter — keep clients with Score_i ≤ T
//
//	Phase C — Per-slice VPSA aggregation (private data access, filtered set):
//	  Step 5: For each collection p, call AggregateSliceVPSA on filtered clients
//	  Step 6: Update global model w_{t+1} = w_t - concat(Δw^(1), ..., Δw^(P))
func (c *VPSAContract) AdaptiveAggregateModelsVPSA(
	ctx contractapi.TransactionContextInterface,
	modelClientIDs []string,
	beta int,
	alpha float64,
) error {
	if len(modelClientIDs) == 0 {
		return fmt.Errorf("no model-client IDs provided")
	}
	if alpha < 0 || alpha > 1 {
		return fmt.Errorf("alpha must be in [0, 1], got %f", alpha)
	}

	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return err
	}
	gm, err := c.GetGlobalModel(ctx)
	if err != nil {
		return err
	}

	globalWeights, err := c.loadGlobalWeightsPrivate(ctx, gm)
	if err != nil {
		return err
	}

	numCollections := len(cfg.Collections)
	if numCollections == 0 {
		return fmt.Errorf("no collections configured")
	}

	n := len(modelClientIDs)

	// Parse client IDs (for reporting)
	clientIDs := make([]string, n)
	for i, mc := range modelClientIDs {
		parts := strings.Split(mc, "::")
		if len(parts) != 2 {
			return fmt.Errorf("invalid modelClientID format: %s", mc)
		}
		clientIDs[i] = parts[1]
	}

	// ══════════════════════════════════════════════════════════════════
	// PHASE A: Per-Slice Partial Anomaly Score Computation
	//
	// For each collection (vertical partition), we:
	//   - Read client slices from ONLY that collection
	//   - Compute coordinate-wise median for that slice
	//   - Accumulate partial scalar summaries per client
	//
	// NO full client vector is EVER assembled.
	// Only scalar partial sums cross collection boundaries.
	// ══════════════════════════════════════════════════════════════════

	// Accumulators for anomaly score partial components (per client)
	partialDistSq := make([]float64, n)  // Σ_p ||G_i^(p) - M^(p)||²
	partialDot := make([]float64, n)     // Σ_p G_i^(p) · M^(p)
	partialNormGSq := make([]float64, n) // Σ_p ||G_i^(p)||²
	partialNormMSq := 0.0                // Σ_p ||M^(p)||²

	// Determine total model size for partition computation
	totalSize := len(globalWeights)
	if totalSize == 0 {
		sampleParts := strings.Split(modelClientIDs[0], "::")
		for _, collName := range cfg.Collections {
			sampleKey := fmt.Sprintf("%s::%s::part", sampleParts[0], sampleParts[1])
			sampleBytes, err := ctx.GetStub().GetPrivateData(collName, sampleKey)
			if err != nil || sampleBytes == nil {
				continue
			}
			sl, _ := floatSliceFromJSONBytes(sampleBytes)
			totalSize += len(sl)
		}
	}
	if totalSize == 0 {
		return fmt.Errorf("cannot determine total model size")
	}

	partitionIndices := paperPartitionIndices(totalSize, numCollections)

	for _, collName := range cfg.Collections {
		// ── Read client slices from THIS collection ONLY ──
		// (Each collection is a separate vertical partition — no cross-collection reads)
		slices := make([][]float64, n)
		for i, mc := range modelClientIDs {
			parts := strings.Split(mc, "::")
			partKey := fmt.Sprintf("%s::%s::part", parts[0], parts[1])
			partBytes, err := ctx.GetStub().GetPrivateData(collName, partKey)
			if err != nil {
				return fmt.Errorf("GetPrivateData failed for %s in %s: %v", mc, collName, err)
			}
			if partBytes == nil {
				return fmt.Errorf("partition missing for %s in %s", mc, collName)
			}
			sliceVec, err := floatSliceFromJSONBytes(partBytes)
			if err != nil {
				return fmt.Errorf("invalid partition JSON for %s in %s: %v", mc, collName, err)
			}
			slices[i] = sliceVec
		}

		sliceLen := len(slices[0])

		// ── Step 1a: Coordinate-wise median for THIS slice ──
		medianSlice := make([]float64, sliceLen)
		for k := 0; k < sliceLen; k++ {
			vals := make([]float64, n)
			for i := 0; i < n; i++ {
				vals[i] = slices[i][k]
			}
			medianSlice[k] = medianFloat64(vals)
		}

		// ── Step 1b: Accumulate partial anomaly scalars ──
		// These are SCALAR summaries — they do NOT reveal individual coordinates.
		for i := 0; i < n; i++ {
			for k := 0; k < sliceLen; k++ {
				diff := slices[i][k] - medianSlice[k]
				partialDistSq[i] += diff * diff                  // partial ||G_i - M||²
				partialDot[i] += slices[i][k] * medianSlice[k]   // partial G_i · M
				partialNormGSq[i] += slices[i][k] * slices[i][k] // partial ||G_i||²
			}
		}
		for k := 0; k < sliceLen; k++ {
			partialNormMSq += medianSlice[k] * medianSlice[k] // partial ||M||²
		}
	}

	// ══════════════════════════════════════════════════════════════════
	// PHASE B: Global Anomaly Score Assembly + Trust-Adjusted Filtering
	//
	// Combine partial scalar summaries into global scores.
	// Only uses scalar values — NO model coordinates are accessed here.
	//
	// D_i  = sqrt( Σ_p distSq_p(i) )
	// C_i  = ( Σ_p dot_p(i) ) / ( sqrt(Σ_p normGSq_p(i)) × sqrt(Σ_p normMSq_p) )
	// Score_i = α × D_i + (1-α) × (1 - C_i)
	//
	// Trust-Adjusted Effective Score:
	//   EffectiveScore_i = Score_i × (2 - TrustScore_i)
	//   Trust=1.0 → no amplification; Trust=0.0 → score doubled
	// ══════════════════════════════════════════════════════════════════

	normM := math.Sqrt(partialNormMSq)

	// Load historical trust scores for all participating clients
	trustRecords := make([]*ClientTrustScore, n)
	for i := 0; i < n; i++ {
		tr, err := c.getClientTrustScore(ctx, clientIDs[i])
		if err != nil {
			return fmt.Errorf("failed to load trust for %s: %v", clientIDs[i], err)
		}
		trustRecords[i] = tr
	}

	scores := make([]float64, n)
	effectiveScores := make([]float64, n)
	for i := 0; i < n; i++ {
		// Global Euclidean distance from partials
		dist := math.Sqrt(partialDistSq[i])

		// Global cosine similarity from partials
		normG := math.Sqrt(partialNormGSq[i])
		var cosSim float64
		if normG > 1e-10 && normM > 1e-10 {
			cosSim = partialDot[i] / (normG * normM)
		} else {
			cosSim = 1.0
		}

		// Raw anomaly score: Score_i = α * D_i + (1-α) * (1 - C_i)
		scores[i] = alpha*dist + (1-alpha)*(1-cosSim)

		// Trust-adjusted effective score:
		// EffectiveScore = RawScore × (2 - Trust)
		// Trust=1.0 → multiplier=1.0 (no change)
		// Trust=0.5 → multiplier=1.5 (50% amplified)
		// Trust=0.0 → multiplier=2.0 (doubled)
		// Capped at 1.5 to prevent vicious feedback loops where
		// a single false positive compounds into permanent rejection.
		trustMultiplier := 2.0 - trustRecords[i].TrustScore
		if trustMultiplier > 1.5 {
			trustMultiplier = 1.5
		}
		effectiveScores[i] = scores[i] * trustMultiplier
	}

	// ── Step 3: IQR-based statistical threshold on EFFECTIVE scores ──
	q1 := percentileFloat64(effectiveScores, 25)
	q3 := percentileFloat64(effectiveScores, 75)
	iqr := q3 - q1

	// Minimum IQR floor: when all honest clients are very similar, the IQR
	// approaches zero, making the threshold too tight and causing false positives.
	// Use 50% of Q3 as a floor to ensure the threshold has reasonable spread.
	minIQR := 0.5 * q3
	if iqr < minIQR {
		iqr = minIQR
	}

	threshold := q3 + 1.5*iqr

	// ── Step 4: Filter malicious updates — keep EffectiveScore_i ≤ T ──
	var filteredMCIDs []string
	var rejectedIDs []string
	clientScores := make(map[string]float64)
	clientEffScores := make(map[string]float64)
	rejectedSet := make(map[string]bool)

	for i := 0; i < n; i++ {
		clientScores[clientIDs[i]] = scores[i]
		clientEffScores[clientIDs[i]] = effectiveScores[i]
		if effectiveScores[i] <= threshold {
			filteredMCIDs = append(filteredMCIDs, modelClientIDs[i])
		} else {
			rejectedIDs = append(rejectedIDs, clientIDs[i])
			rejectedSet[clientIDs[i]] = true
		}
	}

	if len(filteredMCIDs) == 0 {
		return fmt.Errorf("all clients filtered as anomalous — cannot aggregate")
	}

	fmt.Printf("ADAPTIVE VPSA: %d/%d clients passed anomaly filter (threshold=%.6f, alpha=%.2f)\n",
		len(filteredMCIDs), n, threshold, alpha)
	if len(rejectedIDs) > 0 {
		fmt.Printf("ADAPTIVE VPSA: Rejected clients: %v\n", rejectedIDs)
	}

	// ── Step 4b: Update trust scores for ALL participating clients ──
	// Find max raw score for normalization
	maxScore := 0.0
	for i := 0; i < n; i++ {
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}

	clientTrustAfter := make(map[string]float64)
	for i := 0; i < n; i++ {
		tr := trustRecords[i]
		oldTrust := tr.TrustScore
		wasRejected := rejectedSet[clientIDs[i]]

		newTrust := updateTrust(oldTrust, scores[i], maxScore, wasRejected)

		// Record history entry
		entry := TrustRoundEntry{
			Round:        cfg.CurrentRound,
			AnomalyScore: scores[i],
			WasRejected:  wasRejected,
			TrustBefore:  oldTrust,
			TrustAfter:   newTrust,
		}
		tr.TrustScore = newTrust
		tr.RoundsParticipated++
		tr.LastRound = cfg.CurrentRound
		if wasRejected {
			tr.RoundsRejected++
		}
		tr.History = append(tr.History, entry)

		// Write updated trust back to ledger
		if err := c.putClientTrustScore(ctx, tr); err != nil {
			return fmt.Errorf("failed to store trust for %s: %v", clientIDs[i], err)
		}

		clientTrustAfter[clientIDs[i]] = newTrust
		fmt.Printf("  TRUST: %s  %.4f → %.4f  (score=%.6f, rejected=%v)\n",
			clientIDs[i], oldTrust, newTrust, scores[i], wasRejected)
	}

	// ══════════════════════════════════════════════════════════════════
	// PHASE C: Per-Slice VPSA Aggregation on Filtered Set
	//
	// Each collection is aggregated independently using AggregateSliceVPSA.
	// Only filtered (trusted) client IDs are passed.
	// This preserves VPSA's per-slice privacy: no cross-collection access.
	// ══════════════════════════════════════════════════════════════════

	aggregatedDeltas := make([]float64, totalSize)

	for collIdx, collName := range cfg.Collections {
		indices := partitionIndices[collIdx]

		var globalSlice []float64
		if globalWeights != nil && len(globalWeights) == totalSize {
			globalSlice = extractByIndices(globalWeights, indices)
		}

		// AggregateSliceVPSA on filtered clients only — never touches rejected data
		aggDeltaSlice, err := c.AggregateSliceVPSA(ctx, collName, filteredMCIDs, globalSlice, beta)
		if err != nil {
			return fmt.Errorf("AggregateSliceVPSA failed for %s: %v", collName, err)
		}

		scatterByIndices(aggregatedDeltas, indices, aggDeltaSlice)
	}

	// Global model update: w_{t+1} = w_t − Δw_agg
	newGlobal := make([]float64, totalSize)
	if globalWeights != nil && len(globalWeights) == totalSize {
		for i := 0; i < totalSize; i++ {
			newGlobal[i] = globalWeights[i] - aggregatedDeltas[i]
		}
	} else {
		for i := 0; i < totalSize; i++ {
			newGlobal[i] = -aggregatedDeltas[i]
		}
	}

	weightsHash, err := c.storeGlobalWeightsPrivate(ctx, newGlobal)
	if err != nil {
		return err
	}

	ts, _ := getTxTimestamp(ctx)

	gm.Weights = ""
	gm.WeightsHash = weightsHash
	gm.Version++
	gm.Round = cfg.CurrentRound
	gm.Timestamp = ts
	gm.Status = "updated"
	gm.NumClients = len(filteredMCIDs)

	gmBytes, _ := json.Marshal(gm)
	if err := ctx.GetStub().PutState("vpsa-global-model", gmBytes); err != nil {
		return fmt.Errorf("put global model failed: %v", err)
	}

	// Store anomaly report on-chain for auditability (now includes trust scores)
	report := AnomalyReport{
		ReportID:          fmt.Sprintf("anomaly-report-round-%d", cfg.CurrentRound),
		Round:             cfg.CurrentRound,
		Alpha:             alpha,
		Threshold:         threshold,
		Q1:                q1,
		Q3:                q3,
		IQR:               iqr,
		TotalClients:      n,
		FilteredClients:   len(filteredMCIDs),
		RejectedClients:   rejectedIDs,
		ClientScores:      clientScores,
		EffectiveScores:   clientEffScores,
		ClientTrustScores: clientTrustAfter,
		Timestamp:         ts,
		DocType:           "anomalyReport",
	}
	reportBytes, _ := json.Marshal(report)
	_ = ctx.GetStub().PutState(report.ReportID, reportBytes)

	// Mark local models as aggregated
	for _, mc := range filteredMCIDs {
		parts := strings.Split(mc, "::")
		if len(parts) != 2 {
			continue
		}
		metaKey := fmt.Sprintf("%s::%s::meta", parts[0], parts[1])
		metaJSON, _ := ctx.GetStub().GetState(metaKey)
		if metaJSON != nil {
			var meta LocalModelMeta
			_ = json.Unmarshal(metaJSON, &meta)
			meta.Status = "aggregated"
			if b, _ := json.Marshal(meta); b != nil {
				_ = ctx.GetStub().PutState(metaKey, b)
			}
		}
	}

	// Store training metrics
	metrics := TrainingMetrics{
		MetricID:        fmt.Sprintf("metrics-round-%d", cfg.CurrentRound),
		Round:           cfg.CurrentRound,
		GlobalAccuracy:  0.0,
		GlobalLoss:      0.0,
		AlignmentScore:  0.0,
		NumParticipants: len(filteredMCIDs),
		Timestamp:       ts,
	}
	metricsBytes, _ := json.Marshal(metrics)
	_ = ctx.GetStub().PutState(metrics.MetricID, metricsBytes)

	// Advance round
	cfg.CurrentRound++
	cfg.LastUpdated = ts
	cfgBytes, _ := json.Marshal(cfg)
	_ = ctx.GetStub().PutState("vpsa-config", cfgBytes)

	return nil
}

// GetAnomalyReport retrieves the anomaly detection report for a given round
func (c *VPSAContract) GetAnomalyReport(ctx contractapi.TransactionContextInterface, round int) (string, error) {
	reportKey := fmt.Sprintf("anomaly-report-round-%d", round)
	reportBytes, err := ctx.GetStub().GetState(reportKey)
	if err != nil {
		return "", fmt.Errorf("failed to read anomaly report: %v", err)
	}
	if reportBytes == nil {
		return "", fmt.Errorf("anomaly report not found for round %d", round)
	}
	return string(reportBytes), nil
}

// ---------- Trust Score Functions ----------

// getClientTrustScore reads the trust score for a single client from the ledger.
// If no trust record exists, returns a fresh entry with trust=1.0 (fully trusted).
func (c *VPSAContract) getClientTrustScore(ctx contractapi.TransactionContextInterface, clientID string) (*ClientTrustScore, error) {
	key := fmt.Sprintf("trust::%s", clientID)
	b, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, fmt.Errorf("failed to read trust for %s: %v", clientID, err)
	}
	if b == nil {
		// First participation — fully trusted
		return &ClientTrustScore{
			ClientID:           clientID,
			TrustScore:         1.0,
			RoundsParticipated: 0,
			RoundsRejected:     0,
			LastRound:          0,
			History:            []TrustRoundEntry{},
			DocType:            "trustScore",
		}, nil
	}
	var ts ClientTrustScore
	if err := json.Unmarshal(b, &ts); err != nil {
		return nil, fmt.Errorf("unmarshal trust for %s: %v", clientID, err)
	}
	return &ts, nil
}

// putClientTrustScore writes a trust score back to the ledger.
func (c *VPSAContract) putClientTrustScore(ctx contractapi.TransactionContextInterface, ts *ClientTrustScore) error {
	key := fmt.Sprintf("trust::%s", ts.ClientID)
	b, err := json.Marshal(ts)
	if err != nil {
		return fmt.Errorf("marshal trust for %s: %v", ts.ClientID, err)
	}
	return ctx.GetStub().PutState(key, b)
}

// updateTrust computes the new trust score using EMA.
//
//	RoundTrust = 1 - normalizedAnomalyScore  (clamped to [0,1])
//	If rejected:  RoundTrust = 0  (maximum penalty)
//	NewTrust = γ × OldTrust + (1-γ) × RoundTrust
func updateTrust(oldTrust float64, anomalyScore float64, maxScore float64, wasRejected bool) float64 {
	var roundTrust float64
	if wasRejected {
		roundTrust = 0.0
	} else {
		// Normalize anomaly score to [0,1] using max observed score in this round
		if maxScore > 1e-10 {
			normalized := anomalyScore / maxScore
			if normalized > 1.0 {
				normalized = 1.0
			}
			roundTrust = 1.0 - normalized
		} else {
			roundTrust = 1.0
		}
	}
	newTrust := TrustDecayGamma*oldTrust + (1-TrustDecayGamma)*roundTrust
	// Clamp to [0, 1]
	if newTrust < 0 {
		newTrust = 0
	}
	if newTrust > 1 {
		newTrust = 1
	}
	return newTrust
}

// GetClientTrustScore queries the accumulated trust score for a single client.
func (c *VPSAContract) GetClientTrustScore(ctx contractapi.TransactionContextInterface, clientID string) (string, error) {
	ts, err := c.getClientTrustScore(ctx, clientID)
	if err != nil {
		return "", err
	}
	b, _ := json.Marshal(ts)
	return string(b), nil
}

// GetAllTrustScores returns trust scores for all registered clients.
func (c *VPSAContract) GetAllTrustScores(ctx contractapi.TransactionContextInterface) (string, error) {
	clientListJSON, err := ctx.GetStub().GetState("client-list")
	if err != nil {
		return "", fmt.Errorf("failed to read client list: %v", err)
	}
	var clientList []string
	if clientListJSON != nil {
		_ = json.Unmarshal(clientListJSON, &clientList)
	}

	result := make(map[string]*ClientTrustScore)
	for _, cid := range clientList {
		ts, err := c.getClientTrustScore(ctx, cid)
		if err != nil {
			continue
		}
		result[cid] = ts
	}
	b, _ := json.Marshal(result)
	return string(b), nil
}

// ResetAllClientTrust resets trust scores for all registered clients to 1.0.
// This should be called at the start of a new experiment to clear accumulated
// trust from previous runs, preventing false positives from persisted state.
func (c *VPSAContract) ResetAllClientTrust(ctx contractapi.TransactionContextInterface) error {
	clientListJSON, err := ctx.GetStub().GetState("client-list")
	if err != nil {
		return fmt.Errorf("failed to read client list: %v", err)
	}
	var clientList []string
	if clientListJSON != nil {
		_ = json.Unmarshal(clientListJSON, &clientList)
	}

	for _, cid := range clientList {
		key := fmt.Sprintf("trust::%s", cid)
		if err := ctx.GetStub().DelState(key); err != nil {
			return fmt.Errorf("failed to delete trust for %s: %v", cid, err)
		}
	}

	fmt.Printf("TRUST RESET: Cleared trust scores for %d clients\n", len(clientList))
	return nil
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
