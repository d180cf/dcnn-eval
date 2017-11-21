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

const DCNN = require('./dcnn');
const fstext = require('./fstext');
const format = require('./format');

const [, , dcnnFile, inputFiles, weightsBinaryPrecision, searchDepth] = process.argv;

const WEIGHTS_BIN_DIGITS = +weightsBinaryPrecision || 0; // the number of bits per weight
const STEP_DURATION = 1.0; // seconds

if (WEIGHTS_BIN_DIGITS > 0)
    console.log('Weights precision reduced to ' + WEIGHTS_BIN_DIGITS + ' bits');

console.log(`Reconstructing DCNN from ${dcnnFile}`);
const dcnn = new DCNN(JSON.parse(fstext.read(dcnnFile)), WEIGHTS_BIN_DIGITS);

console.log(`Reading SGF files from ${inputFiles}`);
const paths = glob.sync(inputFiles);
console.log((paths.length / 1000).toFixed(0) + ' K files total');
const accuracy = []; // accuracy[asize] = [average, count]
let totalAccuracy = 0;

let t = Date.now(), t0 = t;
let total = 0;

for (const path of paths) {
    const text = fstext.read(path);

    const bsize = +(/\bSZ\[(\d+)\]/.exec(text) || [])[1] || 0;
    const asize = +(/\bAS\[(\d+)\]/.exec(text) || [])[1] || 0;
    const value = +(/\bTS\[(\d+)\]/.exec(text) || [])[1] || 0;

    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const [x, y] = tsumego.stone.coords(target);
    const prediction = deepeval(solver, [x, y], +searchDepth || 0);
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
function deepeval(solver, [x, y], depth = 0) {
    const board = solver.board;
    const hash = board.hash;
    const tblock = board.get(x, y);
    const color = Math.sign(tblock);
    const predictions = [1]; // if no moves available, the group is safe    

    if (!tblock) // the target block has been captured
        return 0;

    if (depth < 1)
        return dcnn.eval(board, [x, y]);

    for (const move of solver.getValidMovesFor(-color)) {
        if (!board.play(move)) // just in case
            continue;

        const responses = [deepeval(solver, [x, y])]; // it's allowed to pass

        for (const resp of solver.getValidMovesFor(color)) {
            if (!board.play(resp))
                continue;

            if (board.hash != hash) { // but not allowed to recapture a ko
                const p = deepeval(solver, [x, y], depth - 1);
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
