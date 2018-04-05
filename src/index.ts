import * as tf from '@tensorflow/tfjs';
import * as types from '../node_modules/@tensorflow/tfjs-layers/dist/types';
import * as coretypes from '../node_modules/@tensorflow/tfjs-core/dist/types';
import { chunk, flatMapDeep, shuffle } from "lodash";

const data = require("./dataset.json");

let wordVecs;
const sequenceLength = 12;
const embeddingDim = 300;
const batchSize = 30;
const filterSizes = [3,4,5];
const numFilters = 256; // 512
const drop = 0.5;
const epochs = 2;
const LEARNING_RATE = 1e-4;

const intents = ["greet", "bye"];
(async () => {
    const resp = await fetch("./word2vec.json");
    wordVecs = await resp.json();
    let model;
    const adam = tf.train.adam(LEARNING_RATE, 0.9, 0.999, 1e-08);
    const optimizer = tf.train.adam(LEARNING_RATE);
    tf.tidy(() => {
        const input = <types.SymbolicTensor>tf.input({ dtype: types.DType.float32, shape: [sequenceLength, embeddingDim, 1] });
        const convLayer1 = tf.layers.conv2d({
            inputShape: [sequenceLength, embeddingDim, 1],
            batchSize,
            kernelSize: [filterSizes[0], embeddingDim],
            filters: numFilters,
            kernelInitializer: 'randomNormal',
            padding: 'valid',
            activation: 'relu',
        });
        const convLayer2 = tf.layers.conv2d({
            inputShape: [sequenceLength, embeddingDim, 1],
            batchSize,
            kernelSize: [filterSizes[1], embeddingDim],
            filters: numFilters,
            kernelInitializer: 'randomNormal',
            padding: 'valid',
            activation: 'relu',
        });
        const convLayer3 = tf.layers.conv2d({
            inputShape: [sequenceLength, embeddingDim, 1],
            batchSize,
            kernelSize: [filterSizes[2], embeddingDim],
            filters: numFilters,
            kernelInitializer: 'randomNormal',
            padding: 'valid',
            activation: 'relu',
        });

        const maxpool1 = <tf.SymbolicTensor>tf.layers.maxPooling2d({
            poolSize: [sequenceLength - filterSizes[0] + 1, 1],
            strides: [1, 1],
            padding: 'valid',
        }).apply(convLayer1.apply(input));

        const maxpool2 = <tf.SymbolicTensor>tf.layers.maxPooling2d({
            poolSize: [sequenceLength - filterSizes[1] + 1, 1],
            strides: [1, 1],
            padding: 'valid',
        }).apply(convLayer2.apply(input));

        const maxpool3 = <tf.SymbolicTensor>tf.layers.maxPooling2d({
            poolSize: [sequenceLength - filterSizes[2] + 1, 1],
            strides: [1, 1],
            padding: 'valid',
        }).apply(convLayer3.apply(input));

        const concatLayer = tf.layers.concatenate({ axis: 1 }).apply([maxpool1, maxpool2, maxpool3]);
        const flat = tf.layers.flatten().apply(concatLayer);
        const dropOut = tf.layers.dropout({ rate: drop }).apply(flat);
        const out = <tf.SymbolicTensor>tf.layers.dense({
            units: intents.length,
            activation: 'softmax'
        }).apply(dropOut);
        model = tf.model({ inputs: input, outputs: out });
        model.compile({
            optimizer: optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });
    });
    const chunks = chunk<IExample>(shuffle(data.examples), batchSize);
    await Promise.all(chunks.map(async (ch) => {
        const res = await trainModel(model, ch);
    }));
    const h = model.history.history;
    const c = h.val_loss.length-1;
    console.log(`Finished Training! => val_loss: ${h.val_loss[c]} | val_acc: ${h.val_acc[c]} | loss: ${h.loss[c]} | acc: ${h.acc[c]}`);
    adam.dispose();
    optimizer.dispose();
    tf.tidy(() => {
        const predictSentence = "good day";
        const predictSentence2 = "leaving";
        const v = [getVectorsFromSentence(predictSentence)];
        const v2 = [getVectorsFromSentence(predictSentence2)];
        const predict = flatMapDeep(v.concat(v2));
        const pre = tf.tensor3d(
            predict, [2, sequenceLength, embeddingDim], types.DType.float32
        ).as4D(2, sequenceLength, embeddingDim, 1);
        const output = <tf.Tensor<tf.Rank>>model.predict(pre);
        const p1 = output.gather(tf.tensor1d([0]), 0);
        const p2 = output.gather(tf.tensor1d([1]), 0);
        p1.print();
        p2.print();
        
        // Dispose
        pre.dispose();
        output.dispose();
        p1.dispose();
        p2.dispose();
        debugger;
    });
    
})();

interface IExample { text: string; intent: string; }
async function trainModel(model: tf.Model, examples: IExample[]) {
    let xs: tf.Tensor4D;
    let ys: tf.Tensor2D; 
    const trainingY = [];
    const trainingX = examples.map(d => {
        trainingY.push(intents.map(k => k === d.intent ? 1 : 0));
        return getVectorsFromSentence(d.text);
    });
    const xts = <coretypes.TensorLike3D>flatMapDeep(trainingX);
    const yts = flatMapDeep(trainingY);
    xs = tf.tensor3d(xts, [examples.length, sequenceLength, embeddingDim])
        .as4D(examples.length, sequenceLength, embeddingDim, 1);
    ys = tf.tensor2d(yts, [examples.length, intents.length]);
    const result = await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: examples.length,
        validationSplit: 0.2,
        // verbose: 1,
    });
    xs.dispose();
    ys.dispose();
}

function getVectorFromWord(word: string) {
    const w = wordVecs[word];
    if (w) { return w; }
    console.warn("Word not found:", word)
    return new Array(embeddingDim).fill(1);
}

function tokenize(sentence: string) {
    return sentence.trim().split(" ").map(w => w.trim()).filter(w => !!w);
}

function getVectorsFromSentence(originalSentence: string, size=sequenceLength) {
    const emptyVector = new Array(embeddingDim).fill(0);
    const ret = new Array(size).fill(emptyVector);
    tokenize(originalSentence).slice(0, size).forEach((w, i) => { ret[i] = getVectorFromWord(w); });
    return ret;
}