import * as tf from '@tensorflow/tfjs';

/**
 * Parse CSV file to extract features and labels
 * Assumes last column is the label (0 or 1 for binary classification)
 * Returns { xs: Tensor2D, ys: Tensor2D }
 */
export async function loadCSVtoTensors(file: File): Promise<{ xs: tf.Tensor2D; ys: tf.Tensor2D }> {
  const text = await file.text();
  const lines = text.trim().split('\n');
  
  // Skip header if present (detect if first row has non-numeric values)
  const startIdx = isNaN(parseFloat(lines[0].split(',')[0])) ? 1 : 0;
  
  const data: number[][] = [];
  const labels: number[] = [];
  
  for (let i = startIdx; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => parseFloat(v.trim()));
    if (values.some(isNaN)) continue; // Skip invalid rows
    
    // Last column is label
    const label = values.pop()!;
    labels.push(label);
    data.push(values);
  }
  
  const xs = tf.tensor2d(data);
  const ys = tf.tensor2d(labels, [labels.length, 1]);
  
  return { xs, ys };
}

/**
 * Parse comma-separated string into array of floats
 */
export function parseQueryVector(input: string): number[] {
  return input.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
}

/**
 * Generate example dataset for binary classification
 * Returns { xs: Tensor2D, ys: Tensor2D }
 */
export function generateExampleData(numSamples = 100, numFeatures = 8): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const data: number[][] = [];
  const labels: number[] = [];
  
  for (let i = 0; i < numSamples; i++) {
    const features: number[] = [];
    let sum = 0;
    
    for (let j = 0; j < numFeatures; j++) {
      const val = Math.random();
      features.push(val);
      sum += val;
    }
    
    // Simple rule: if average > 0.5, label = 1, else 0
    const label = sum / numFeatures > 0.5 ? 1 : 0;
    
    data.push(features);
    labels.push(label);
  }
  
  const xs = tf.tensor2d(data);
  const ys = tf.tensor2d(labels, [labels.length, 1]);
  
  return { xs, ys };
}

/**
 * Normalize tensor (z-score normalization)
 */
export function normalizeTensor(tensor: tf.Tensor2D): { normalized: tf.Tensor2D; mean: tf.Tensor1D; std: tf.Tensor1D } {
  const mean = tensor.mean(0) as tf.Tensor1D;
  const std = tf.moments(tensor, 0).variance.sqrt() as tf.Tensor1D;
  
  // Avoid division by zero
  const stdSafe = std.add(1e-7);
  const normalized = tensor.sub(mean).div(stdSafe) as tf.Tensor2D;
  
  return { normalized, mean, std };
}

/**
 * Train a simple logistic regression model for binary classification
 * @param xs - Input features (Tensor2D)
 * @param ys - Labels (Tensor2D)
 * @param epochs - Number of training epochs
 * @param onEpoch - Callback function called after each epoch with (epoch, loss, accuracy)
 * @returns Trained tf.LayersModel
 */
export async function trainModel(
  xs: tf.Tensor2D,
  ys: tf.Tensor2D,
  epochs: number = 50,
  onEpoch?: (epoch: number, logs: { loss: number; acc: number }) => void
): Promise<tf.LayersModel> {
  const inputShape = xs.shape[1];
  
  // Build simple logistic regression model
  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        inputShape: [inputShape],
        kernelInitializer: 'randomNormal',
        biasInitializer: 'zeros'
      })
    ]
  });
  
  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  // Train model
  await model.fit(xs, ys, {
    epochs,
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (onEpoch && logs) {
          onEpoch(epoch, {
            loss: logs.loss as number,
            acc: logs.acc as number
          });
        }
      }
    }
  });
  
  return model;
}

/**
 * Flatten all model weights into a single 1D array
 * @param model - Trained TensorFlow.js model
 * @returns Float32Array of all weights concatenated
 */
export function flattenWeights(model: tf.LayersModel): number[] {
  const weights = model.getWeights();
  const arrays: Float32Array[] = [];
  
  // Synchronously extract data from each tensor
  for (const tensor of weights) {
    arrays.push(tensor.dataSync() as Float32Array);
  }
  
  // Calculate total length
  const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);
  
  // Concatenate all arrays
  const flattened = new Float32Array(totalLength);
  let offset = 0;
  
  for (const arr of arrays) {
    flattened.set(arr, offset);
    offset += arr.length;
  }
  
  // Convert to regular JS array
  return Array.from(flattened);
}

/**
 * Split array into N equal parts or by specified sizes
 * @param arr - Input array
 * @param sizesOrN - Either number of parts (N) or array of sizes [size1, size2, ...]
 * @returns Array of subarrays
 */
export function splitArray(arr: number[], sizesOrN: number | number[]): number[][] {
  if (typeof sizesOrN === 'number') {
    // Split into N equal parts
    const n = sizesOrN;
    const partSize = Math.ceil(arr.length / n);
    const parts: number[][] = [];
    
    for (let i = 0; i < n; i++) {
      const start = i * partSize;
      const end = Math.min(start + partSize, arr.length);
      parts.push(arr.slice(start, end));
    }
    
    return parts;
  } else {
    // Split by specified sizes
    const sizes = sizesOrN;
    const parts: number[][] = [];
    let offset = 0;
    
    for (const size of sizes) {
      parts.push(arr.slice(offset, offset + size));
      offset += size;
    }
    
    return parts;
  }
}

/**
 * Split weights into exactly 2 parts for private data collections
 * @param weights - Flattened weight array
 * @returns Object with collectionOrg1Private and collectionOrg2Private
 */
export function splitWeightsForCollections(weights: number[]): {
  collectionOrg1Private: number[];
  collectionOrg2Private: number[];
} {
  const [part1, part2] = splitArray(weights, 2);
  
  return {
    collectionOrg1Private: part1,
    collectionOrg2Private: part2
  };
}

/**
 * Calculate model accuracy on test data
 */
export async function evaluateModel(
  model: tf.LayersModel,
  xs: tf.Tensor2D,
  ys: tf.Tensor2D
): Promise<{ loss: number; accuracy: number }> {
  const result = model.evaluate(xs, ys) as tf.Scalar[];
  const loss = await result[0].data();
  const accuracy = await result[1].data();
  
  return {
    loss: loss[0],
    accuracy: accuracy[0]
  };
}