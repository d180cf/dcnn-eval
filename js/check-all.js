const fstext = require('./fstext');
const fspath = require('path');
const glob = require('glob');
const tsumego = require('tsumego.js');

const [, , inputFiles, numTests] = process.argv;

const paths = glob.sync(inputFiles);

console.log(`checking ${numTests} out of ${paths.length} problems...`);

let npassed = 0;
let nfailed = 0;
let nsolved = 0;

for (let i = 0; i < +numTests; i++) {
    const path = paths[Math.random() * paths.length | 0];
    const text = fstext.read(path);
    const status = +/\bTS\[(.)\]/.exec(text)[1];
    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const color = tsumego.sign(solver.board.get(solver.target));
    const player = +/\bDS\[(.)\]/.exec(text)[1] ? +color : -color;
    const bestmove = (/\bTR\[(..)\]/.exec(text) || [])[1];
    const komaster = -color;
    const move = solver.solve(player, komaster);
    const safe = move * color > 0;

    try {
        if (safe != !!status)
            throw 'wrong status (1)';

        if ((move * player > 0) != !!bestmove)
            throw 'wrong status (2)';

        if (bestmove) {
            const move2 = (player > 0 ? 'B' : 'W') + '[' + bestmove + ']';

            if (!board.play(tsumego.stone.fromString(move2)))
                throw 'invalid best move ' + bestmove;

            if (board.get(target)) {
                const resp2 = solver.solve(-player, komaster);

                if (resp2 * player < 0)
                    throw `invalid solution (defense is ${tsumego.stone.toString(resp2)})`;
            }

            nsolved++;
        }

        npassed++;
    } catch (reason) {
        nfailed++;
        console.log(reason, path, tsumego.stone.toString(move));
    }
}

console.log(`passed = ${npassed}; failed = ${nfailed}; solved = ${nsolved}`);
