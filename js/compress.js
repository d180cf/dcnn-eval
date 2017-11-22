/**
 * Reduces precision of NN weights.
 * Weights are expected to be in the -1..1 range.
 */

const fstext = require('./fstext');

const [, , inputPath, outputPath, _precision] = process.argv;

const precision = +_precision;
const multiplier = 1 << precision - 1; // 1 bit is reserved for the sign

console.log('Input JSON:', inputPath);
console.log('Output JSON:', outputPath);
console.log('Precision:', precision, 'bits per weight');

const input = fstext.read(inputPath);
const json = JSON.parse(input);

console.log('Original size:', (input.length / 2 ** 20).toFixed(1), 'MB');

for (const name in json.vars) {
    const item = json.vars[name];
    const data = item.data;

    console.log(name, data.length, 'weights');

    for (let i = 0; i < data.length; i++) {
        const x = data[i];
        const y = Math.round(x * multiplier) / multiplier;

        data[i] = Math.min(1, Math.max(-1, y)); // fit into -1..1 range
    }
}

const output = JSON.stringify(json);
fstext.write(outputPath, output);
console.log('Compressed size:', (output.length / 2 ** 20).toFixed(1), 'MB');
