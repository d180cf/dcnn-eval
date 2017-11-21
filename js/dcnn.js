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

const [, , dcnnFile, inputFiles, searchDepth] = process.argv;

const WINDOW_SIZE = 11; // 11x11, must match the DCNN
const WINDOW_HALF = WINDOW_SIZE / 2 | 0;
const STEP_DURATION = 1.0; // seconds

console.log(`Reconstructing DCNN from ${dcnnFile}`);
const evalDCNN = reconstructDCNN(JSON.parse(fstext.read(dcnnFile)))

console.log(`Reading SGF files from ${inputFiles}`);
const paths = glob.sync(inputFiles);
console.log((paths.length / 1000).toFixed(0) + ' K files total');
const accuracy = []; // accuracy[asize] = [average, count]
let totalAccuracy = 0;

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
    const prediction = evaluate(solver, [x, y], +searchDepth || 0);
    const iscorrect = (value - 0.5) * (prediction - 0.5) > 0;

    totalAccuracy = (totalAccuracy * total + (iscorrect ? 1 : 0)) / (total + 1);

    const [average, n] = accuracy[asize] || [0, 0];

    accuracy[asize] = [
        (average * n + (iscorrect ? 1 : 0)) / (n + 1),
        n + 1];

    total++;

    // report progress in a fancy way
    if (Date.now() > t + STEP_DURATION * 1000 || total == paths.length) {
        t = Date.now();

        const done = total / paths.length; // 0..1
        const speed = total / (t - t0) * 1000; // SGF files per second

        const length = 40;
        const len1 = done * length | 0;
        const len2 = length - len1;

        process.stdout.write([
            (' ' + (done * 100 | 0)).slice(-3) + '%',
            '\u2588'.repeat(len1) + '\u2592'.repeat(len2),
            (totalAccuracy.toFixed(2) + '0').slice(0, 4),
            speed.toFixed(0) + ' N/s'
        ].join(' ') + '\r');

        if (total == paths.length)
            process.stdout.write('\n');
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

console.log('Error by area size:\n');
console.log(format('{0:5} | {1:5} | {2:5}', 'size', 'error', 'count'));
console.log(format('{0:5} | {1:5} | {2:5}', '-'.repeat(5), '-'.repeat(5), '-'.repeat(5)));
let asum = 0;

for (let asize = 0; asize < accuracy.length; asize++) {
    const [average, n] = accuracy[asize] || [0, 0];

    if (n > 0) {
        console.log(format('{0:5} | {1:5} | {2:5}', asize, (1 - average).toFixed(3), n));
        asum += n * average;
    }
}

console.log('\nAverage: ' + (1 - asum / total).toFixed(3));

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
 * @param {JSON} json description of the NN
 */
function reconstructDCNN(json) {
    const input = nn.value([WINDOW_SIZE * WINDOW_SIZE * F_COUNT]);

    /*

    (1573, 256) align/dense/weights:0
         (256,) align/dense/bias:0

     (256, 256) resb1/1/dense/weights:0
         (256,) resb1/1/dense/bias:0

     (256, 256) resb1/2/dense/weights:0
         (256,) resb1/2/dense/bias:0

       (256, 1) readout/dense/weights:0
           (1,) readout/dense/bias:0

    */

    function get(name) {
        const v = json.vars[name];
        return v && v.data;
    }

    function fconn(x, w, b) {
        x = nn.mul(x, w);
        x = nn.add(x, b);
        return x;
    }

    let x = input;

    // the alignment layer
    x = fconn(x,
        get('align/dense/weights:0'),
        get('align/dense/bias:0'));
    x = nn.relu(x);

    // the residual tower
    for (let i = 1; ; i++) {
        const w_name = k => `resb${i}/${k}/dense/weights:0`;
        const b_name = k => `resb${i}/${k}/dense/bias:0`;

        const w1 = get(w_name(1));
        const b1 = get(b_name(1));

        const w2 = get(w_name(2));
        const b2 = get(b_name(2));

        if (!w1) break;

        const y = x;

        x = fconn(x, w1, b1);
        x = nn.relu(x);

        x = fconn(x, w2, b2);
        x = nn.add(x, y);
        x = nn.relu(x);
    }

    // the readout layer
    x = fconn(x,
        get('readout/dense/weights:0'),
        get('readout/dense/bias:0'));
    x = nn.sigmoid(x);

    if (x.size != 1)
        throw Error('Invalid output shape: ' + x.shape);

    return data => {
        input.set(data);
        x.eval();
        return x.value[0];
    };
}
