package main

import (
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
	Weights          string  `json:"weights"` // JSON string or base64; we will parse when needed
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
		Weights:          "[]",
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
	return &gm, nil
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
	if err := json.Unmarshal(tBytes, &payload); err != nil {
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
		if err := ctx.GetStub().PutPrivateData(collName, partKey, partBytes); err != nil {
			return fmt.Errorf("PutPrivateData failed (%s): %v", collName, err)
		}
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

// fetchPartitionsForModel collects partitions for a given modelID across all configured collections.
// It returns map[clientID] -> concatenated vector ([]float64) assembled by concatenating partitions in collection order.
// It discovers clients by reading "client-list" and checking for model meta keys: "<modelID>::<clientID>::meta".
func (c *VPSAContract) fetchPartitionsForModel(ctx contractapi.TransactionContextInterface, modelID string) (map[string][]float64, error) {
	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return nil, err
	}

	// read registered clients
	clientListJSON, err := ctx.GetStub().GetState("client-list")
	if err != nil {
		return nil, fmt.Errorf("failed to read client-list: %v", err)
	}
	var clientList []string
	if clientListJSON != nil {
		if err := json.Unmarshal(clientListJSON, &clientList); err != nil {
			return nil, fmt.Errorf("invalid client-list json: %v", err)
		}
	}

	result := make(map[string][]float64)

	// For each registered client, check if this client has submitted meta for this model
	for _, clientID := range clientList {
		metaKey := fmt.Sprintf("%s::%s::meta", modelID, clientID)
		metaBytes, err := ctx.GetStub().GetState(metaKey)
		if err != nil {
			return nil, fmt.Errorf("failed to read meta for %s: %v", metaKey, err)
		}
		if metaBytes == nil {
			// client didn't submit this model
			continue
		}

		// assemble full vector by concatenating partitions in configured collection order
		var fullVec []float64
		for _, coll := range cfg.Collections {
			partKey := fmt.Sprintf("%s::%s::part", modelID, clientID)
			partBytes, err := ctx.GetStub().GetPrivateData(coll, partKey)
			if err != nil {
				return nil, fmt.Errorf("GetPrivateData failed for coll %s key %s: %v", coll, partKey, err)
			}
			if partBytes == nil {
				// missing partition for this collection -> treat as error (you can change to skip if desired)
				return nil, fmt.Errorf("private part missing for coll %s key %s", coll, partKey)
			}
			partVec, err := floatSliceFromJSONBytes(partBytes)
			if err != nil {
				return nil, fmt.Errorf("invalid partition json for client %s coll %s: %v", clientID, coll, err)
			}
			fullVec = append(fullVec, partVec...)
		}

		result[clientID] = fullVec
	}

	return result, nil
}

// AggregateModelsVPSA implements the VPSA aggregation by reading private data partitions from configured collections
// NOTE: modelIDs parameter must be provided as strings of the form "<modelID>::<clientID>" to allow chaincode to find private data parts.
// For example: ["m1::clientA", "m1::clientB"]
func (c *VPSAContract) AggregateModelsVPSA(ctx contractapi.TransactionContextInterface, modelClientIDs []string, beta int) error {
	if len(modelClientIDs) == 0 {
		return fmt.Errorf("no model-client ids provided")
	}

	cfg, err := c.GetAggregationConfig(ctx)
	if err != nil {
		return err
	}
	gm, err := c.GetGlobalModel(ctx)
	if err != nil {
		return err
	}

	// Parse global weights (assume stored as JSON []float64 in gm.Weights)
	var globalWeights []float64
	if err := json.Unmarshal([]byte(gm.Weights), &globalWeights); err != nil {
		// if empty or not parseable, we will treat globalWeights as zeros
		globalWeights = nil
	}

	// Build per-client full vectors by concatenating partitions in configured collection order.
	type clientVec struct {
		clientID string
		vec      []float64
		domain   string
	}
	var cvs []clientVec

	for _, mc := range modelClientIDs {
		// expect format modelID::clientID
		parts := strings.Split(mc, "::")
		if len(parts) != 2 {
			return fmt.Errorf("modelClientIDs must be of form modelID::clientID; got %s", mc)
		}
		modelID := parts[0]
		clientID := parts[1]
		// confirm client exists
		clientJSON, err := ctx.GetStub().GetState(clientID)
		if err != nil || clientJSON == nil {
			return fmt.Errorf("client %s not found", clientID)
		}
		var client Client
		_ = json.Unmarshal(clientJSON, &client)

		var fullVec []float64
		for _, coll := range cfg.Collections {
			partKey := fmt.Sprintf("%s::%s::part", modelID, clientID)
			partBytes, err := ctx.GetStub().GetPrivateData(coll, partKey)
			if err != nil {
				return fmt.Errorf("GetPrivateData failed for col %s key %s: %v", coll, partKey, err)
			}
			if partBytes == nil {
				return fmt.Errorf("private part missing for coll %s key %s", coll, partKey)
			}
			partVec, err := floatSliceFromJSONBytes(partBytes)
			if err != nil {
				return fmt.Errorf("invalid partition json: %v", err)
			}
			fullVec = append(fullVec, partVec...)
		}
		cvs = append(cvs, clientVec{
			clientID: clientID,
			vec:      fullVec,
			domain:   client.Domain,
		})
	}

	numClients := len(cvs)
	if numClients == 0 {
		return fmt.Errorf("no client vectors found")
	}
	vecLen := len(cvs[0].vec)
	// sanity: ensure all vectors same length
	for _, cv := range cvs {
		if len(cv.vec) != vecLen {
			return fmt.Errorf("vector length mismatch among clients")
		}
	}

	// Compute cosine weights per client (cosine between client's vector and globalWeights)
	cosW := make([]float64, numClients)
	for i, cv := range cvs {
		if globalWeights != nil && len(globalWeights) == vecLen {
			cs, err := cosineSimilarity(cv.vec, globalWeights)
			if err != nil {
				// if similarity can't compute, fallback
				cs = 0
			}
			// transform cosine to positive weight: (1+cos)/2 to map [-1,1] -> [0,1]
			cosW[i] = (1.0 + cs) / 2.0
		} else {
			// if no global anchor, default uniform
			cosW[i] = 1.0
		}
	}

	// Prepare a 2D slice to access coordinate k across clients: vals[k][i]
	// but to save memory we compute per-coordinate streaming
	aggNumerator := make([]float64, vecLen)
	aggDenom := make([]float64, vecLen)

	// For each coordinate, gather values across clients, sort and trim according to beta
	for k := 0; k < vecLen; k++ {
		// collect values and pairs (value, index)
		type pair struct {
			val   float64
			index int
		}
		arr := make([]pair, numClients)
		for i := 0; i < numClients; i++ {
			arr[i] = pair{val: cvs[i].vec[k], index: i}
		}
		// sort by value
		sort.Slice(arr, func(i, j int) bool { return arr[i].val < arr[j].val })

		// compute euclidean-based mask (we will zero weights for trimmed entries)
		// The paper uses coordinate-wise trimming: remove top beta and bottom beta values.
		trimLow := 0
		trimHigh := 0
		if beta > 0 {
			trimLow = beta
			trimHigh = beta
			if trimLow*2 >= numClients {
				// can't trim more than available
				trimLow = 0
				trimHigh = 0
			}
		}

		// compute final weighted sum for this coordinate
		var numSum float64
		var denSum float64
		for idx := 0; idx < numClients; idx++ {
			// arr idx in sorted order; check if trimmed
			trimmed := false
			if idx < trimLow || idx >= (numClients-trimHigh) {
				trimmed = true
			}
			i := arr[idx].index // original client index
			if trimmed {
				// zero out (skip)
				continue
			}
			weight := cosW[i] // combine with euclidean weight which is 1 here unless you add further logic
			// if you want to multiply by local euclidean distance-based weight, compute here
			numSum += weight * arr[idx].val
			denSum += weight
		}
		aggNumerator[k] = numSum
		aggDenom[k] = denSum
	}

	// Build aggregated delta vector (Δw)
	delta := make([]float64, vecLen)
	for k := 0; k < vecLen; k++ {
		delta[k] = safeDiv(aggNumerator[k], aggDenom[k])
	}

	// Now update global model: w_{t+1} = w_t - delta (paper's style). If gm.Weights empty, initialize with zeros
	if globalWeights == nil || len(globalWeights) != vecLen {
		globalWeights = make([]float64, vecLen)
		for i := 0; i < vecLen; i++ {
			globalWeights[i] = 0.0
		}
	}

	newGlobal := make([]float64, vecLen)
	for i := 0; i < vecLen; i++ {
		newGlobal[i] = globalWeights[i] - delta[i]
	}

	// Serialize new global weights and write to world state
	newGlobalBytes, err := json.Marshal(newGlobal)
	if err != nil {
		return fmt.Errorf("marshal new global failed: %v", err)
	}
	gm.Weights = string(newGlobalBytes)
	gm.Version++
	gm.Round = cfg.CurrentRound
	gm.Timestamp, _ = getTxTimestamp(ctx)
	gm.Status = "updated"

	if gmB, err := json.Marshal(gm); err == nil {
		if err := ctx.GetStub().PutState("vpsa-global-model", gmB); err != nil {
			return fmt.Errorf("put global failed: %v", err)
		}
	} else {
		return fmt.Errorf("marshal gm failed: %v", err)
	}

	// mark local partitions as aggregated: put status meta
	for _, mc := range modelClientIDs {
		parts := strings.Split(mc, "::")
		if len(parts) != 2 {
			continue
		}
		modelID := parts[0]
		clientID := parts[1]
		metaKey := fmt.Sprintf("%s::%s::meta", modelID, clientID)
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

	// store training metrics
	metrics := TrainingMetrics{
		MetricID:        fmt.Sprintf("metrics-round-%d", cfg.CurrentRound),
		Round:           cfg.CurrentRound,
		GlobalAccuracy:  0.0,
		GlobalLoss:      0.0,
		AlignmentScore:  0.0,
		NumParticipants: len(modelClientIDs),
		Timestamp:       gm.Timestamp,
	}
	if mb, err := json.Marshal(metrics); err == nil {
		_ = ctx.GetStub().PutState(metrics.MetricID, mb)
	}

	// advance round and persist config
	cfg.CurrentRound++
	cfg.LastUpdated = gm.Timestamp
	if cj, err := json.Marshal(cfg); err == nil {
		_ = ctx.GetStub().PutState("vpsa-config", cj)
	}

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

	// create additive shares such that sum(shares) = queryVec
	// simple approach: shares[0..n-2] random small numbers, last share = query - sum
	// In chaincode we cannot create secure randomness across orgs; clients normally create shares.
	// We'll implement deterministic pseudo-splitting for demonstration:
	shares := make([][]float64, n)
	for i := 0; i < n; i++ {
		shares[i] = make([]float64, len(queryVec))
	}
	// first n-1 shares: take fractional portions
	for i := 0; i < len(queryVec); i++ {
		val := queryVec[i]
		portion := val / float64(n) // equal split
		for s := 0; s < n-1; s++ {
			shares[s][i] = portion
		}
		shares[n-1][i] = val - portion*float64(n-1)
	}

	// store shares into private collections with key "query::<queryID>::share::<i>"
	for i, coll := range cfg.Collections {
		key := fmt.Sprintf("query::%s::share::%d", queryID, i)
		b, _ := json.Marshal(shares[i])
		if err := ctx.GetStub().PutPrivateData(coll, key, b); err != nil {
			return fmt.Errorf("put private share failed: %v", err)
		}
	}

	// client/peers will then perform local inference on their share and return logits into another private key
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

// ---------- main ----------

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
