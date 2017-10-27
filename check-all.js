const fstext = require('./fstext');
const fspath = require('path');
const glob = require('glob');
const tsumego = require('tsumego.js');

const [, , inputFiles, probability] = process.argv;

const paths = glob.sync(inputFiles);

console.log(`checking ${probability * 100 | 0} % out of ${paths.length} problems...`);

let npassed = 0;
let nfailed = 0;

for (const path of paths) {
    if (Math.random() > probability)
        continue;

    const text = fstext.read(path);
    const status = +/\bTS\[(.)\]/.exec(text)[1];
    const solver = new tsumego.Solver(text);
    const color = tsumego.sign(solver.board.get(solver.target));
    const move = solver.solve(-color, -color);
    const safe = tsumego.sign(move) * color > 0;

    if (safe == !!status) {
        npassed++;
    } else {
        nfailed++;
        console.log(`wrong status: ${path} ${tsumego.stone.toString(move)}`);
    }
}

console.log(`passed = ${npassed}; failed = ${nfailed}`);
