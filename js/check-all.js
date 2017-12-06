const fstext = require('./fstext');
const fspath = require('path');
const glob = require('glob');
const tsumego = require('tsumego.js');

const [, , inputFiles, numTests] = process.argv;

const paths = glob.sync(inputFiles);

console.log(`checking ${numTests} out of ${paths.length} problems...`);

let npassed = 0;
let nfailed = 0;

for (let i = 0; i < +numTests; i++) {
    const path = paths[Math.random() * paths.length | 0];
    const text = fstext.read(path);
    const status = +/\bTS\[(.)\]/.exec(text)[1];
    const solver = new tsumego.Solver(text);
    const color = tsumego.sign(solver.board.get(solver.target));
    const player = +/\bDS\[(.)\]/.exec(text)[1] ? +color : -color;
    const komaster = -color;
    const move = solver.solve(player, komaster);
    const safe = tsumego.sign(move) * color > 0;

    if (safe == !!status) {
        npassed++;
    } else {
        nfailed++;
        console.log(`wrong status: ${path} ${tsumego.stone.toString(move)}`);
    }
}

console.log(`passed = ${npassed}; failed = ${nfailed}`);
