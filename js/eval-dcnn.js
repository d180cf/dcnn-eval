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

const [, , dcnnFile, inputFiles, maxDuration] = process.argv;

console.log(`Reconstructing DCNN from ${dcnnFile}`);
const dcnn = new DCNN(JSON.parse(fstext.read(dcnnFile)));

console.log(`Reading SGF files from ${inputFiles}`);
const paths = glob.sync(inputFiles);
console.log((paths.length / 1000).toFixed(0) + ' K files total');
const accuracy = []; // accuracy[asize] = [average, count]
let totalAccuracy = 0;
let total = 0;

const ends = Date.now() + (+maxDuration) * 1000;

while (Date.now() < ends) {
    const path = paths[Math.random() * paths.length | 0];
    const text = fstext.read(path);

    const bsize = +/\bSZ\[(\d+)\]/.exec(text)[1];
    const asize = +/\bAS\[(\d+)\]/.exec(text)[1];
    const value = +/\bTS\[(\d+)\]/.exec(text)[1];
    const dfndr = +/\bDS\[(\d+)\]/.exec(text)[1];

    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const [x, y] = tsumego.stone.coords(target);
    const prediction = dcnn.eval(board, [x, y], dfndr);
    const iscorrect = (value - 0.5) * (prediction - 0.5) > 0;

    totalAccuracy = (totalAccuracy * total + (iscorrect ? 1 : 0)) / (total + 1);

    const [average, n] = accuracy[asize] || [0, 0];

    accuracy[asize] = [
        (average * n + (iscorrect ? 1 : 0)) / (n + 1),
        n + 1];

    total++;
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
