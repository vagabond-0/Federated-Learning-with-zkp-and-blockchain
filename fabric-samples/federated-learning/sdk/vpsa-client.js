'use strict';

const { connectGateway, getContract } = require('./connection');
const zlib = require('zlib');

/**
 * VPSA Fabric Client
 * Handles all interactions with the VPSA chaincode including transient data
 */
class VPSAClient {
    constructor() {
        this.gateway = null;
        this.client = null;
        this.contract = null;
    }

    /**
     * Connect to the Fabric network
     */
    async connect() {
        console.log('Connecting to Fabric network...');
        const { gateway, client } = await connectGateway();
        this.gateway = gateway;
        this.client = client;
        this.contract = await getContract(gateway);
        console.log('Connected successfully');
    }

    /**
     * Disconnect from the network
     */
    disconnect() {
        if (this.gateway) {
            this.gateway.close();
        }
        if (this.client) {
            this.client.close();
        }
        console.log('Disconnected from Fabric network');
    }

    /**
     * Initialize the ledger
     */
    async initLedger() {
        console.log('Initializing ledger...');
        await this.contract.submitTransaction('InitLedger');
        console.log('Ledger initialized');
    }

    /**
     * Register a new client
     */
    async registerClient(clientID, domain, datasetSize) {
        console.log(`Registering client: ${clientID}, domain: ${domain}`);
        await this.contract.submitTransaction(
            'RegisterClient',
            clientID,
            domain,
            datasetSize.toString()
        );
        console.log(`Client ${clientID} registered successfully`);
    }

    /**
     * Submit local model using standard method (weights as JSON strings)
     */
    async submitLocalModel(modelID, clientID, weights, latentFeatures, prototypes,
        accuracy, loss, alignmentLoss, dataSize) {
        
        console.log(`Submitting local model: ${modelID}`);
        
        const weightsJSON = typeof weights === 'string' ? weights : JSON.stringify(weights);
        const latentJSON = typeof latentFeatures === 'string' ? latentFeatures : JSON.stringify(latentFeatures);
        const protoJSON = typeof prototypes === 'string' ? prototypes : JSON.stringify(prototypes);

        await this.contract.submitTransaction(
            'SubmitLocalModel',
            modelID,
            clientID,
            weightsJSON,
            latentJSON,
            protoJSON,
            accuracy.toString(),
            loss.toString(),
            alignmentLoss.toString(),
            dataSize.toString()
        );
        
        console.log(`Model ${modelID} submitted successfully`);
    }

    /**
     * Submit local model using transient data (private, compressed)
     * This is the recommended method for large models
     * 
     * @param {string} modelID - Unique model identifier
     * @param {string} clientID - Client identifier
     * @param {number} round - Training round number
     * @param {string} dataHash - Hash of the model data for verification
     * @param {Object} modelPayload - The model data object containing:
     *   - weights: map of layer names to weight arrays
     *   - latentFeatures: map of latent feature vectors
     *   - prototypes: map of prototype vectors
     *   - accuracy: model accuracy
     *   - loss: model loss
     *   - alignmentLoss: alignment loss
     *   - dataSize: size of training data
     */
    async submitLocalModelWithTransient(modelID, clientID, round, dataHash, modelPayload) {
        console.log(`Submitting model with transient data: ${modelID}`);
        console.log(`  - Payload size: ${JSON.stringify(modelPayload).length} bytes`);

        // Convert payload to JSON
        const payloadJSON = JSON.stringify(modelPayload);
        
        // Compress with gzip
        const compressedPayload = zlib.gzipSync(Buffer.from(payloadJSON));
        console.log(`  - Compressed size: ${compressedPayload.length} bytes`);
        console.log(`  - Compression ratio: ${(compressedPayload.length / payloadJSON.length * 100).toFixed(1)}%`);

        // Create the transient data map
        const transientData = {
            model: compressedPayload
        };

        // Submit transaction with transient data
        const proposal = this.contract.newProposal('SubmitLocalModelWithTransient', {
            arguments: [modelID, clientID, round.toString(), dataHash],
            transientData: transientData
        });

        const transaction = await proposal.endorse();
        const result = await transaction.submit();
        
        const timestamp = new TextDecoder().decode(result);
        console.log(`Model ${modelID} submitted at: ${timestamp}`);
        
        return timestamp;
    }

    /**
     * Aggregate models using private data
     */
    async aggregateModelsWithPrivateData(modelIDs) {
        console.log(`Aggregating ${modelIDs.length} models...`);
        
        const modelIDsJSON = JSON.stringify(modelIDs);
        const resultBytes = await this.contract.submitTransaction(
            'AggregateModelsWithPrivateData',
            modelIDsJSON
        );
        
        const result = JSON.parse(new TextDecoder().decode(resultBytes));
        console.log('Aggregation result:', result);
        
        return result;
    }

    /**
     * Get global model
     */
    async getGlobalModel() {
        const resultBytes = await this.contract.evaluateTransaction('GetGlobalModel');
        return JSON.parse(new TextDecoder().decode(resultBytes));
    }

    /**
     * Get aggregation config
     */
    async getAggregationConfig() {
        const resultBytes = await this.contract.evaluateTransaction('GetAggregationConfig');
        return JSON.parse(new TextDecoder().decode(resultBytes));
    }

    /**
     * Get local model metadata
     */
    async getLocalModel(modelID) {
        const resultBytes = await this.contract.evaluateTransaction('GetLocalModel', modelID);
        return JSON.parse(new TextDecoder().decode(resultBytes));
    }

    /**
     * Get training metrics for a round
     */
    async getTrainingMetrics(round) {
        const resultBytes = await this.contract.evaluateTransaction(
            'GetTrainingMetrics',
            round.toString()
        );
        return JSON.parse(new TextDecoder().decode(resultBytes));
    }

    /**
     * Get all training metrics
     */
    async getAllTrainingMetrics() {
        const resultBytes = await this.contract.evaluateTransaction('GetAllTrainingMetrics');
        return JSON.parse(new TextDecoder().decode(resultBytes));
    }

    /**
     * Get all clients
     */
    async getAllClients() {
        const resultBytes = await this.contract.evaluateTransaction('GetAllClients');
        return JSON.parse(new TextDecoder().decode(resultBytes));
    }

    /**
     * Update aggregation config
     */
    async updateAggregationConfig(minClients, sourceWeight, targetWeight, alignmentWeight) {
        await this.contract.submitTransaction(
            'UpdateAggregationConfig',
            minClients.toString(),
            sourceWeight.toString(),
            targetWeight.toString(),
            alignmentWeight.toString()
        );
        console.log('Aggregation config updated');
    }
}

module.exports = VPSAClient;