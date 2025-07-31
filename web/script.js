// AI Poetry Generator with ONNX.js
let model = null;
let metadata = null;

// Set seed text examples
function setSeed(text) {
    document.getElementById('seedText').value = text;
}

// Load metadata
async function loadMetadata() {
    try {
        const response = await fetch('../models/model_metadata.json');
        metadata = await response.json();
        console.log('Metadata loaded:', metadata);
    } catch (error) {
        console.error('Error loading metadata:', error);
    }
}

// Load ONNX model
async function loadModel(modelName) {
    try {
        const modelPath = `../models/${modelName}_text_generator.onnx`;
        model = await ort.InferenceSession.create(modelPath);
        console.log(`Model ${modelName} loaded`);
        return true;
    } catch (error) {
        console.error(`Error loading model ${modelName}:`, error);
        return false;
    }
}

// Generate text
async function generateText(seedText, length, temperature) {
    if (!model || !metadata) {
        throw new Error('Model or metadata not loaded');
    }
    
    // Encode starting text
    const charToIdx = metadata.char_to_idx;
    const idxToChar = metadata.idx_to_char;
    
    let generated = seedText;
    let inputSequence = [];
    
    // Convert seed to indices
    for (let char of seedText) {
        if (char in charToIdx) {
            inputSequence.push(charToIdx[char]);
        }
    }
    
    // Initialize hidden state
    const batchSize = 1;
    const hiddenSize = metadata.hidden_size;
    const selectedModel = document.getElementById('modelSelect').value;
    
    // Get num_layers with proper fallback for each model
    let numLayers = 1; // default to 1
    if (metadata.model_info && metadata.model_info[selectedModel] && metadata.model_info[selectedModel].num_layers) {
        numLayers = metadata.model_info[selectedModel].num_layers;
    } else {
        // Fallback values based on our model configuration
        if (selectedModel === 'RNN' || selectedModel === 'GRU') {
            numLayers = 1; // Both RNN and GRU have 1 layer
        } else if (selectedModel === 'LSTM') {
            numLayers = 2; // LSTM has 2 layers
        }
    }
    
    // Initialize hidden states
    let hidden_h = null;
    let hidden_c = null;
    let hidden = null;
    
    if (selectedModel === 'LSTM') {
        hidden_h = new ort.Tensor('float32', new Float32Array(numLayers * batchSize * hiddenSize), [numLayers, batchSize, hiddenSize]);
        hidden_c = new ort.Tensor('float32', new Float32Array(numLayers * batchSize * hiddenSize), [numLayers, batchSize, hiddenSize]);
    } else {
        hidden = new ort.Tensor('float32', new Float32Array(numLayers * batchSize * hiddenSize), [numLayers, batchSize, hiddenSize]);
    }
    
    // Generate character by character
    for (let i = 0; i < length; i++) {
        // Prepare input
        const input = new ort.Tensor('int64', new BigInt64Array([BigInt(inputSequence[inputSequence.length - 1])]), [1, 1]);
        
        // Prepare feeds based on model type
        let feeds = { 'input_sequence': input };
        
        if (selectedModel === 'LSTM') {
            feeds['hidden_h'] = hidden_h;
            feeds['hidden_c'] = hidden_c;
        } else {
            feeds['hidden'] = hidden;
        }
        
        // Inference
        const results = await model.run(feeds);
        
        // Update hidden states for next iteration
        if (selectedModel === 'LSTM') {
            hidden_h = results.new_hidden_h;
            hidden_c = results.new_hidden_c;
        } else {
            hidden = results.new_hidden;
        }
        
        // Get output
        const output = results.output.data;
        
        // Apply temperature and sample
        const vocabSize = metadata.vocab_size;
        const logits = Array.from(output.slice(-vocabSize));
        
        // Apply temperature
        const scaledLogits = logits.map(x => x / temperature);
        
        // Softmax
        const maxLogit = Math.max(...scaledLogits);
        const expLogits = scaledLogits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probabilities = expLogits.map(x => x / sumExp);
        
        // Sample
        const random = Math.random();
        let cumsum = 0;
        let nextCharIdx = 0;
        
        for (let j = 0; j < probabilities.length; j++) {
            cumsum += probabilities[j];
            if (random <= cumsum) {
                nextCharIdx = j;
                break;
            }
        }
        
        // Add generated character
        const nextChar = idxToChar[nextCharIdx.toString()];
        generated += nextChar;
        
        // Update state
        inputSequence = [nextCharIdx];
    }
    
    return generated;
}

// Main generation function
async function generatePoem() {
    const button = document.getElementById('generateBtn');
    const output = document.getElementById('output');
    const modelSelect = document.getElementById('modelSelect');
    const seedText = document.getElementById('seedText').value || 'The ';
    const length = parseInt(document.getElementById('length').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    
    try {
        button.disabled = true;
        button.textContent = 'Generating...';
        output.textContent = 'Loading model and generating poetry...';
        
        // Load model if necessary
        const modelName = modelSelect.value;
        const loaded = await loadModel(modelName);
        
        if (!loaded) {
            throw new Error(`Could not load model ${modelName}`);
        }
        
        // Generate text
        const generatedText = await generateText(seedText, length, temperature);
        
        // Display result
        output.textContent = generatedText;
        
    } catch (error) {
        console.error('Generation error:', error);
        output.textContent = `Error: ${error.message}\n\nNote: Make sure ONNX models are available in the 'models/' folder.`;
    } finally {
        button.disabled = false;
        button.textContent = 'Generate Poem';
    }
}

// Update displayed values
document.getElementById('length').addEventListener('input', function() {
    document.getElementById('lengthValue').textContent = this.value;
});

document.getElementById('temperature').addEventListener('input', function() {
    document.getElementById('tempValue').textContent = this.value;
});

// Load metadata on startup
window.addEventListener('load', loadMetadata);