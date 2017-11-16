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
 */

const fs = require('fs');
const fspath = require('path');
const glob = require('glob');
const tsumego = require('tsumego.js');

const nn = require('./nn');
const fstext = require('./fstext');
const format = require('./format');
const { features, F_COUNT } = require('./features');

const [, , dcnnFile, inputFiles] = process.argv;

const WINDOW_SIZE = 11; // 11x11, must match the DCNN
const WINDOW_HALF = WINDOW_SIZE / 2 | 0;
const STEP_DURATION = 1.0; // seconds

console.log(`Reconstructing DCNN from ${dcnnFile}`);
const evalDCNN = reconstructDCNN(JSON.parse(fstext.read(dcnnFile)))

console.log(`Reading SGF files from ${inputFiles}`);
const paths = glob.sync(inputFiles);
console.log(paths.length + ' files total');
const accuracy = []; // accuracy[asize] = [average, count]

let t = Date.now(), t0 = t;
let total = 0;

const planes = new Float32Array(18 * 18 * F_COUNT); // enough for any board size
const fslice = new Float32Array(WINDOW_SIZE * WINDOW_SIZE * F_COUNT); // no need to recreate it

for (const path of paths) {
    const text = fstext.read(path);

    const bsize = +(/\bSZ\[(\d+)\]/.exec(text) || [])[1] || 0;
    const asize = +(/\bAS\[(\d+)\]/.exec(text) || [])[1] || 0;
    const value = +(/\bTS\[(\d+)\]/.exec(text) || [])[1] || 0;

    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const [x, y] = tsumego.stone.coords(target);
    const prediction = evaluate(solver, [x, y], 1);
    const iscorrect = (value - 0.5) * (prediction - 0.5) > 0;

    const [average, n] = accuracy[asize] || [0, 0];

    accuracy[asize] = [
        (average * n + (iscorrect ? 1 : 0)) / (n + 1),
        n + 1];

    total++;

    // report progress in a fancy way
    if (Date.now() > t + STEP_DURATION * 1000) {
        t = Date.now();

        const done = total / paths.length; // 0..1
        const speed = total / (t - t0) * 1000; // SGF files per second
        const remaining = (paths.length - total) / speed; // seconds

        const length = 40;
        const len1 = done * length | 0;
        const len2 = length - len1;

        const progress = '█'.repeat(len1) + '▒'.repeat(len2);
        const percentage = (' ' + (done * 100 | 0)).slice(-2) + '%';
        const rem_ms = (' ' + (remaining / 60 | 0) + ':' + (remaining | 0) % 60).slice(-5);
        const eoln = total == paths.length ? '\n' : '\r';

        process.stdout.write(percentage + ' ' + progress + ' ' + rem_ms + eoln);
    }
}

/**
 * @param {tsumego.Solver} board 
 */
function evaluate(solver, [x, y], depth = 0) {
    const board = solver.board;
    const hash = board.hash;
    const tblock = board.get(x, y);
    const color = Math.sign(tblock);
    const predictions = [1]; // if no moves available, the group is safe    

    if (!tblock) // the target block has been captured
        return 0;

    if (depth < 1) {
        planes.fill(0); // just in case

        features(planes, board, { x, y });

        // (x + 1, y + 1) is to account for the wall
        slice(fslice, planes, [board.size + 2, board.size + 2, F_COUNT],
            [y + 1 - WINDOW_HALF, y + 1 + WINDOW_HALF],
            [x + 1 - WINDOW_HALF, x + 1 + WINDOW_HALF],
            [0, F_COUNT - 1]);

        return evalDCNN(fslice);
    }

    for (const move of solver.getValidMovesFor(-color)) {
        if (!board.play(move)) // just in case
            continue;

        const responses = [evaluate(solver, [x, y])]; // it's allowed to pass

        for (const resp of solver.getValidMovesFor(color)) {
            if (!board.play(resp))
                continue;

            if (board.hash != hash) { // but not allowed to recapture a ko
                const p = evaluate(solver, [x, y], depth - 1);
                responses.push(p);
            }

            board.undo();
        }

        predictions.push(Math.max(...responses)); // it's trying to save the group
        board.undo();
    }

    return Math.min(...predictions); // it's trying to capture the group
}

console.log('Accuracy by area size:');
let asum = 0;

for (let asize = 0; asize < accuracy.length; asize++) {
    const [average, n] = accuracy[asize] || [0, 0];

    if (n > 0) {
        console.log(format('{0:2} {1:4} {2:4}', asize, average.toFixed(2), n));
        asum += n * average;
    }
}

console.log('Average: ' + (asum / total).toFixed(2));

function print(name, size, depth, data) {
    console.log(name + ' = [ // ' + size + 'x' + size + 'x' + depth);
    for (let d = 0; d < depth; d++) {
        console.log('  [ // feature = ' + d);
        for (let y = 0; y < size; y++) {
            let s = '';
            for (let x = 0; x < size; x++)
                s += (data[y * size * depth + x * depth + d] ? '+' : '-') + ' ';
            console.log('    ' + s.slice(0, -1));
        }
        console.log('  ],');
    }
    console.log(']');
}

function offsetFn(x_size, y_size, z_size) {
    return (x, y, z) => (x * y_size + y) * z_size + z;
}

function slice(res, src, [x_size, y_size, z_size], [xmin, xmax], [ymin, ymax], [zmin, zmax]) {
    const ires = offsetFn(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const isrc = offsetFn(x_size, y_size, z_size);

    for (let x = xmin; x <= xmax; x++) {
        for (let y = ymin; y <= ymax; y++) {
            for (let z = zmin; z <= zmax; z++) {
                const ri = ires(x - xmin, y - ymin, z - zmin);
                const si = isrc(x, y, z);
                const outside = x < 0 || x >= x_size || y < 0 || y >= y_size || z < 0 || z >= z_size;

                res[ri] = outside ? 0 : src[si];
            }
        }
    }
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
    const input = nn.value([WINDOW_SIZE * WINDOW_SIZE * F_COUNT]);

    let x = input;

    for (let i = 0; ; i++) {
        const w = json.vars['internal/weights:' + i];
        const b = json.vars['internal/bias:' + i];

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

    if (x.size != 1)
        throw Error('Invalid output shape: ' + x.shape);

    return data => {
        input.set(data);
        x.eval();
        return x.value[0];
    };
}
