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

console.log('Evaluation stops in ' + maxDuration + 's');
console.log(`Reconstructing DCNN from ${dcnnFile}`);
const dcnn = new DCNN(JSON.parse(fstext.read(dcnnFile)));

console.log(`Reading SGF files from ${inputFiles}`);
const paths = glob.sync(inputFiles);
console.log((paths.length / 1000).toFixed(0) + ' K files total');
const accuracy = []; // accuracy[asize] = [average, count]
let totalAccuracy = 0;
let total = 0;

const nwmbyas = []; // number of wrong moves by area size

const time = Date.now();
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
    const tcolor = Math.sign(board.get(target));
    const [tx, ty] = tsumego.stone.coords(target);
    const [p_value, p_moves] = dcnn.eval(board, [tx, ty], dfndr);
    const iscorrect = (value - 0.5) * (p_value - 0.5) > 0;

    if (dfndr == value) {
        // console.log(path);
        // console.log(value, p_value);
        // console.log('\n' + board.text + '\n');

        const indexes = [...range(0, p_moves.length - 1)].sort((i, j) => p_moves[j] - p_moves[i]);

        for (let i = 0; i < indexes.length - 1; i++)
            if (p_moves[indexes[i]] < p_moves[indexes[i + 1]])
                throw Error('Unsorted');

        let nwrongs = 0;

        for (let i = 0; i < 5; i++) {
            const j = indexes[i];
            const p = p_moves[j];
            const n = Math.round(Math.sqrt(p_moves.length));
            const y = (j / n | 0) - (n / 2 | 0) + ty;
            const x = (j % n | 0) - (n / 2 | 0) + tx;
            const c = dfndr ? +tcolor : -tcolor;
            const m = tsumego.stone.make(x, y, c);
            const s = (() => {
                if (!board.play(m))
                    return 'invalid';
                const r = board.get(target) ?
                    solver.solve(-c, -tcolor) :
                    -tcolor;
                board.undo();
                return r * c > 0 ? 'correct' : 'wrong';
            })();

            // console.log(tsumego.stone.toString(m) + ' ' + p.toFixed(4) + ' ' + s);

            if (s == 'correct')
                break;

            if (s == 'wrong')
                nwrongs++;
        }

        // console.log(value.toFixed(2) + ' ' + p_value.toFixed(2) + ' ' + nwrongs + ' ' + asize);

        nwmbyas[asize] = nwmbyas[asize] || [];
        nwmbyas[asize].push(nwrongs);
    }

    totalAccuracy = (totalAccuracy * total + (iscorrect ? 1 : 0)) / (total + 1);

    const [average, n] = accuracy[asize] || [0, 0];

    accuracy[asize] = [
        (average * n + (iscorrect ? 1 : 0)) / (n + 1),
        n + 1];

    total++;
}

console.log(`Speed: ${total / (Date.now() - time) * 1000 | 0} N/s`);
console.log('Error by area size:\n');
console.log(format('{0:5} | {1:5} | {2:5} | {3:5}', 'size', 'error', 'wrongs', 'count'));
console.log(format('{0:5} | {1:5} | {2:5} | {3:5}', '-'.repeat(5), '-'.repeat(5), '-'.repeat(5), '-'.repeat(5)));
let asum = 0;

for (let asize = 0; asize < accuracy.length; asize++) {
    const [average, n] = accuracy[asize] || [0, 0];

    const anwms = avg(nwmbyas[asize] || []);

    if (n > 0) {
        console.log(format('{0:5} | {1:5} | {2:5} | {3:5}', asize, (1 - average).toFixed(3), anwms.toFixed(3), n));
        asum += n * average;
    }
}

console.log('\nAverage: ' + (1 - asum / total).toFixed(3));

function* range(min, max) {
    for (let i = min; i <= max; i++)
        yield i;
}

function avg(a, n = a.length) {
    let s = 0;

    for (let i = 0; i < n; i++)
        s += a[i];

    return s / n;
}