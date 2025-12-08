'use strict';

const VPSAClient = require('./vpsa-client');
const crypto = require('crypto');

/**
 * Generate mock model weights for testing
 */
function generateMockWeights(layerName, size) {
    const weights = {};
    weights[layerName] = Array.from({ length: size }, () => Math.random() * 2 - 1);
    return weights;
}

/**
 * Generate a complete mock model payload
 */
function generateMockModelPayload(clientID, accuracy, dataSize) {
    return {
        weights: {
            'encoder.layer1': Array.from({ length: 768 }, () => Math.random() * 0.1),
            'encoder.layer2': Array.from({ length: 768 }, () => Math.random() * 0.1),
            'decoder.layer1': Array.from({ length: 512 }, () => Math.random() * 0.1),
        },
        latentFeatures: {
            'z_mean': Array.from({ length: 256 }, () => Math.random()),
            'z_var': Array.from({ length: 256 }, () => Math.random() * 0.5),
        },
        prototypes: {
            'class_0': Array.from({ length: 128 }, () => Math.random()),
            'class_1': Array.from({ length: 128 }, () => Math.random()),
            'class_2': Array.from({ length: 128 }, () => Math.random()),
        },
        accuracy: accuracy,
        loss: 1.0 - accuracy + Math.random() * 0.1,
        alignmentLoss: Math.random() * 0.2,
        dataSize: dataSize
    };
}

/**
 * Main test function
 */
async function main() {
    const client = new VPSAClient();

    try {
        // Connect to network
        await client.connect();

        console.log('\n========================================');
        console.log('VPSA CHAINCODE TEST WITH TRANSIENT DATA');
        console.log('========================================\n');

        // Test 1: Get initial global model
        console.log('--- Test 1: Get Global Model ---');
        const globalModel = await client.getGlobalModel();
        console.log('Global Model:', JSON.stringify(globalModel, null, 2));

        // Test 2: Get config
        console.log('\n--- Test 2: Get Aggregation Config ---');
        const config = await client.getAggregationConfig();
        console.log('Config:', JSON.stringify(config, null, 2));

        // Test 3: Register clients
        console.log('\n--- Test 3: Register Clients ---');
        try {
            await client.registerClient('client-source-1', 'source', 5000);
            await client.registerClient('client-source-2', 'source', 4500);
            await client.registerClient('client-target-1', 'target', 3000);
            await client.registerClient('client-target-2', 'target', 2500);
            console.log('Clients registered successfully');
        } catch (err) {
            console.log('Clients may already exist:', err.message);
        }

        // Test 4: Get all clients
        console.log('\n--- Test 4: Get All Clients ---');
        const clients = await client.getAllClients();
        console.log('Registered clients:', clients.length);
        clients.forEach(c => console.log(`  - ${c.clientID} (${c.domain}): ${c.datasetSize} samples`));

        // Test 5: Submit models with transient data
        console.log('\n--- Test 5: Submit Models with Transient Data ---');
        
        const modelsToSubmit = [
            { id: 'model-r0-src1', client: 'client-source-1', accuracy: 0.82, dataSize: 5000 },
            { id: 'model-r0-src2', client: 'client-source-2', accuracy: 0.79, dataSize: 4500 },
            { id: 'model-r0-tgt1', client: 'client-target-1', accuracy: 0.71, dataSize: 3000 },
            { id: 'model-r0-tgt2', client: 'client-target-2', accuracy: 0.68, dataSize: 2500 },
        ];

        const submittedModelIDs = [];

        for (const model of modelsToSubmit) {
            const payload = generateMockModelPayload(model.client, model.accuracy, model.dataSize);
            const dataHash = crypto.createHash('sha256')
                .update(JSON.stringify(payload))
                .digest('hex')
                .substring(0, 16);

            console.log(`\nSubmitting ${model.id}...`);
            const timestamp = await client.submitLocalModelWithTransient(
                model.id,
                model.client,
                0, // round 0
                dataHash,
                payload
            );
            console.log(`  Submitted at: ${timestamp}`);
            submittedModelIDs.push(model.id);
        }

        // Wait for transactions to commit
        console.log('\nWaiting for transactions to commit...');
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Test 6: Verify submitted models
        console.log('\n--- Test 6: Verify Submitted Models ---');
        for (const modelID of submittedModelIDs) {
            try {
                const modelMeta = await client.getLocalModel(modelID);
                console.log(`  ${modelID}: status=${modelMeta.status}, accuracy=${modelMeta.accuracy}`);
            } catch (err) {
                console.log(`  ${modelID}: Not found (may use -meta suffix)`);
            }
        }

        // Test 7: Aggregate models
        console.log('\n--- Test 7: Aggregate Models with Private Data ---');
        const aggregationResult = await client.aggregateModelsWithPrivateData(submittedModelIDs);
        console.log('Aggregation Result:', JSON.stringify(aggregationResult, null, 2));

        // Test 8: Get updated global model
        console.log('\n--- Test 8: Get Updated Global Model ---');
        const updatedGlobalModel = await client.getGlobalModel();
        console.log('Updated Global Model:', JSON.stringify(updatedGlobalModel, null, 2));

        // Test 9: Get training metrics
        console.log('\n--- Test 9: Get Training Metrics ---');
        try {
            const metrics = await client.getTrainingMetrics(0);
            console.log('Round 0 Metrics:', JSON.stringify(metrics, null, 2));
        } catch (err) {
            console.log('Metrics not available:', err.message);
        }

        console.log('\n========================================');
        console.log('ALL TESTS COMPLETED SUCCESSFULLY!');
        console.log('========================================\n');

    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    } finally {
        client.disconnect();
    }
}

main();