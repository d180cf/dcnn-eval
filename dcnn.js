/**
 * Applies the trained DCNN.
 * 
 * Input:
 * 
 *      1. JSON file with DCNN weights.
 *      2. SGF file with the tsumego.
 * 
 * Output:
 * 
 *      1. Probablity that the target group is safe.
 * 
 */

const fs = require('fs');
const fspath = require('path');
const glob = require('glob');
const tsumego = require('tsumego.js');

const nn = require('./nn');
const fstext = require('./fstext');
const format = require('./format');
const { features } = require('./features');

const [, , dcnnFile, inputFiles] = process.argv;

const WINDOW_SIZE = 11; // 11x11, must match the DCNN
const WINDOW_HALF = WINDOW_SIZE / 2 | 0;
const STEP_DURATION = 5.0; // seconds

console.log(`Reconstructing DCNN from ${dcnnFile}`);
const evalDCNN = reconstructDCNN(JSON.parse(fstext.read(dcnnFile)))

console.log(`Reading SGF files from ${inputFiles}`);
const paths = glob.sync(inputFiles);
const accuracy = []; // accuracy[asize] = [average, count]

let t = Date.now(), t0 = t;
let total = 0;

for (const path of paths) {
    const text = fstext.read(path);

    const asize = +(/\bAS\[(\d+)\]/.exec(text) || [])[1] || 0;
    const value = +(/\bTS\[(\d+)\]/.exec(text) || [])[1] || 0;

    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const [x, y] = tsumego.stone.coords(target);
    const planes = features(board, { x, y });
    const input = slice(planes,
        [x - WINDOW_HALF, x + WINDOW_HALF],
        [y - WINDOW_HALF, y + WINDOW_HALF]);

    const prediction = evalDCNN(flatten(input));
    const iscorrect = (value - 0.5) * (prediction - 0.5) > 0;

    const [average, n] = accuracy[asize] || [0, 0];

    accuracy[asize] = [
        (average * n + (iscorrect ? 1 : 0)) / (n + 1),
        n + 1];

    total++;

    if (Date.now() > t + STEP_DURATION * 1000) {
        t = Date.now();
        console.log(`${(total / paths.length).toFixed(2)} files processed, ${(total / (t - t0)).toFixed(1)} K/s`);
    }
}

console.log('Accuracy by area size:');
for (let asize = 0; asize < accuracy.length; asize++) {
    const [average, n] = accuracy[asize] || [0, 0];
    n && console.log(format('{0:2} {1:4} {2:4}', asize, average.toFixed(2), n));
}

function flatten(x) {
    while (x[0].length)
        x = [].concat(...x);
    return x;
}

function slice(input, [xmin, xmax], [ymin, ymax]) {
    const [zmin, zmax] = [0, input[0][0].length - 1];
    const output = [];

    for (let x = xmin; x <= xmax; x++) {
        for (let y = ymin; y <= ymax; y++) {
            for (let z = zmin; z <= zmax; z++) {
                // this chunk of code simply does this:
                //
                //  output[x - xmin][y - ymin][z - zmin]
                //      = input[x][y][z]

                const _x = x - xmin;
                const _y = y - ymin;
                const _z = z - zmin;

                let t = output;

                t = t[_x] = t[_x] || [];
                t = t[_y] = t[_y] || [];

                let s = input;

                s = s && s[x] || 0;
                s = s && s[y] || 0;
                s = s && s[z] || 0;

                t[_z] = s;
            }
        }
    }

    return output;
}

/**
 * Reconstructs the DCNN that correponds
 * to the weights file. The result is a 
 * function that takes a tensor as input
 * and returns a single value in the 0..1
 * range.
 * 
 * @param {JSON} json weights of the DCNN
 * @returns {(tensor: number[]) => number}
 */
function reconstructDCNN(json) {
    return data => {
        if (data.length != WINDOW_SIZE * WINDOW_SIZE * 5)
            throw Error('Invalid input size: ' + data.length);

        let x = data;

        for (let i = 1; ; i++) {
            const w = json.vars[`internal-${i}/weights:0`];
            const b = json.vars[`internal-${i}/bias:0`];

            if (!w || !b) break;

            x = nn.mul(x, w.data);
            x = nn.add(x, b.data);
            x = nn.relu(x);
        }

        const w = json.vars['readout/weights:0'];
        const b = json.vars['readout/bias:0'];

        x = nn.mul(x, w.data);
        x = nn.add(x, b.data);
        x = nn.sigmoid(x);

        return x;
    };
}
