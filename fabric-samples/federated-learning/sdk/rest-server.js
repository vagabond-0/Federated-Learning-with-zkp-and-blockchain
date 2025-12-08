'use strict';

const express = require('express');
const bodyParser = require('body-parser');
const VPSAClient = require('./vpsa-client');
const zlib = require('zlib');

const app = express();
const PORT = 3000;

// Increase payload limit for large models
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.raw({ limit: '50mb', type: 'application/octet-stream' }));

let fabricClient = null;

/**
 * Initialize Fabric connection
 */
async function initFabric() {
    fabricClient = new VPSAClient();
    await fabricClient.connect();
    console.log('Fabric client connected');
}

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
    res.json({ status: 'ok', connected: fabricClient !== null });
});

/**
 * Initialize ledger
 */
app.post('/api/init', async (req, res) => {
    try {
        await fabricClient.initLedger();
        res.json({ success: true, message: 'Ledger initialized' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Register a new FL client
 */
app.post('/api/clients', async (req, res) => {
    try {
        const { clientID, domain, datasetSize } = req.body;
        
        if (!clientID || !domain || !datasetSize) {
            return res.status(400).json({ 
                success: false, 
                error: 'Missing required fields: clientID, domain, datasetSize' 
            });
        }
        
        await fabricClient.registerClient(clientID, domain, datasetSize);
        res.json({ success: true, message: `Client ${clientID} registered` });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Get all registered clients
 */
app.get('/api/clients', async (req, res) => {
    try {
        const clients = await fabricClient.getAllClients();
        res.json({ success: true, clients });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Submit local model with transient data (compressed)
 * Expects JSON body with model payload
 */
app.post('/api/models/submit', async (req, res) => {
    try {
        const { modelID, clientID, round, dataHash, payload } = req.body;
        
        if (!modelID || !clientID || round === undefined || !payload) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: modelID, clientID, round, payload'
            });
        }
        
        console.log(`Receiving model ${modelID} from ${clientID}`);
        console.log(`  Payload keys: ${Object.keys(payload)}`);
        
        const timestamp = await fabricClient.submitLocalModelWithTransient(
            modelID,
            clientID,
            round,
            dataHash || '',
            payload
        );
        
        res.json({ 
            success: true, 
            message: `Model ${modelID} submitted`,
            timestamp 
        });
    } catch (error) {
        console.error('Submit error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Submit local model (standard method, for smaller models)
 */
app.post('/api/models/submit-standard', async (req, res) => {
    try {
        const { 
            modelID, clientID, weights, latentFeatures, prototypes,
            accuracy, loss, alignmentLoss, dataSize 
        } = req.body;
        
        await fabricClient.submitLocalModel(
            modelID, clientID, weights, latentFeatures, prototypes,
            accuracy, loss, alignmentLoss, dataSize
        );
        
        res.json({ success: true, message: `Model ${modelID} submitted` });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Get local model metadata
 */
app.get('/api/models/:modelID', async (req, res) => {
    try {
        const model = await fabricClient.getLocalModel(req.params.modelID);
        res.json({ success: true, model });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Aggregate models
 */
app.post('/api/aggregate', async (req, res) => {
    try {
        const { modelIDs } = req.body;
        
        if (!modelIDs || !Array.isArray(modelIDs)) {
            return res.status(400).json({
                success: false,
                error: 'Missing required field: modelIDs (array)'
            });
        }
        
        console.log(`Aggregating ${modelIDs.length} models...`);
        const result = await fabricClient.aggregateModelsWithPrivateData(modelIDs);
        res.json({ success: true, result });
    } catch (error) {
        console.error('Aggregation error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Get global model
 */
app.get('/api/global-model', async (req, res) => {
    try {
        const model = await fabricClient.getGlobalModel();
        res.json({ success: true, model });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Get aggregation config
 */
app.get('/api/config', async (req, res) => {
    try {
        const config = await fabricClient.getAggregationConfig();
        res.json({ success: true, config });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Update aggregation config
 */
app.put('/api/config', async (req, res) => {
    try {
        const { minClients, sourceWeight, targetWeight, alignmentWeight } = req.body;
        await fabricClient.updateAggregationConfig(
            minClients, sourceWeight, targetWeight, alignmentWeight
        );
        res.json({ success: true, message: 'Config updated' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Get training metrics for a round
 */
app.get('/api/metrics/:round', async (req, res) => {
    try {
        const metrics = await fabricClient.getTrainingMetrics(parseInt(req.params.round));
        res.json({ success: true, metrics });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Get all training metrics
 */
app.get('/api/metrics', async (req, res) => {
    try {
        const metrics = await fabricClient.getAllTrainingMetrics();
        res.json({ success: true, metrics });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

/**
 * Start the server
 */
async function start() {
    try {
        await initFabric();
        app.listen(PORT, () => {
            console.log(`VPSA REST API server running on http://localhost:${PORT}`);
            console.log('Available endpoints:');
            console.log('  GET  /health');
            console.log('  POST /api/init');
            console.log('  POST /api/clients');
            console.log('  GET  /api/clients');
            console.log('  POST /api/models/submit');
            console.log('  POST /api/models/submit-standard');
            console.log('  GET  /api/models/:modelID');
            console.log('  POST /api/aggregate');
            console.log('  GET  /api/global-model');
            console.log('  GET  /api/config');
            console.log('  PUT  /api/config');
            console.log('  GET  /api/metrics/:round');
            console.log('  GET  /api/metrics');
        });
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

// Handle shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down...');
    if (fabricClient) {
        fabricClient.disconnect();
    }
    process.exit(0);
});

start();