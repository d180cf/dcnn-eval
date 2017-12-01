const fstext = require('./fstext');
const fspath = require('path');
const glob = require('glob');
const tsumego = require('tsumego.js');

const [, , inputFiles, maxCount] = process.argv;

const paths = glob.sync(inputFiles);

console.log(`checking ${maxCount} out of ${paths.length} problems...`);

let npassed = 0;
let nfailed = 0;

for (let i = 0; i < maxCount; i++) {
    const path = paths[Math.random() * paths.length | 0];
    const text = fstext.read(path);
    const label = /\bTS\[(.)\]/.exec(text)[1];
    const solver = new tsumego.Solver(text);
    const color = tsumego.sign(solver.board.get(solver.target));
    const move1 = solver.solve(+color, -color); // defense
    const move2 = solver.solve(-color, -color); // attack
    const status = move1 * move2 < 0 ? 0 : Math.sign(move1) * color;
    const mstr = tsumego.stone.toString;

    if (label == ['-', '=', '+'][status + 1]) {
        npassed++;
    } else {
        nfailed++;
        console.log(`wrong status: ${path} ${label} defense=${mstr(move1)} attack=${mstr(move2)}`);
    }
}

console.log(`passed = ${npassed}; failed = ${nfailed}`);
