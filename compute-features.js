/**
 * Usage: node parse-sgf bin/json 2
 * 
 * Input: SGF files from the `sgf-problems` module.
 * Output: JSON files with features and labels.
 */

if (process.argv.length < 4) {
    console.log('Missing script arguments');
    process.exit(0);
}

const fs = require('fs');
const fspath = require('path');
const mkdirp = require('mkdirp');
const sgf = require('sgf-problems');
const tsumego = require('tsumego.js');

const resdir = process.argv[2];
const fpsize = +process.argv[3];

for (const dir of sgf.dirs) {
    for (const file of dir.problems) {
        console.log('[-] ' + file.path);
        const sgf = file + '';
        
        try {
            const solver = new tsumego.Solver(sgf);
            const board = solver.board;
            const color = tsumego.sign(board.get(solver.target));
            console.log('[?] solving the problem...');
            const move = solver.solve(-color, -color);
            const safe = tsumego.stone.color(move) * -color > 0 ? 0 : 1;            
            const [x, y] = tsumego.stone.coords(solver.target);
            console.log('[?] computing the features...');
            const feat = features(board, { x, y }, fpsize);

            const json = JSON.stringify({
                label: safe ? 1 : 0, // safe means the target cannot be captured
                features: feat
            });

            const respath = fspath.join(resdir, file.path).replace(/\.sgf$/, '.json');
            mkdirp.sync(fspath.dirname(respath));
            fs.writeFileSync(respath, json, 'utf8');
            console.log('[+] ' + respath);
        } catch (err) {
            console.log('[!] ' + err);
        }
    }
}

/**
 * Computes features of the given board
 * and returns them as a list of feature
 * planes where each number is in `0..1` range.
 * 
 * @param {tsumego.Board} board
 * @param {{x: number, y: number}} target
 * @param {number} size
 * @returns {number[][][]}
 */
function features(board, target, size) {
    const result = tensor([5, size * 2 + 1, size * 2 + 1]);

    const FI_A = 0;
    const FI_E = 1;
    const FI_N = 2;
    const FI_1 = 3;
    const FI_S = 4;

    const targetColor = tsumego.sign(board.get(target.x, target.y));

    if (!targetColor)
        throw Error('The target location is empty');

    for (let x = target.x - size; x <= target.x + size; x++) {
        for (let y = target.y - size; y <= target.y + size; y++) {
            const i = x - (target.x - size);
            const j = y - (target.y - size);

            if (!board.inBounds(x, y)) {
                result[FI_A][i][j] = 0;
                result[FI_E][i][j] = 0;
                result[FI_N][i][j] = 1;
                result[FI_1][i][j] = 0;
                result[FI_S][i][j] = 0;
            } else {
                const block = board.get(x, y);
                const nlibs = tsumego.block.libs(block);
                const nsize = tsumego.block.size(block);

                result[FI_A][i][j] = block * targetColor > 0 ? 1 : 0;
                result[FI_E][i][j] = block * targetColor < 0 ? 1 : 0;
                result[FI_N][i][j] = 0;
                result[FI_1][i][j] = nlibs == 1 ? 1 : 0;
                result[FI_S][i][j] = nsize > 1 ? 1 : 0;
            }
        }
    }

    return result;
}

/**
 * Creates a tensor of the given shape.
 * 
 *  - `tensor([]) = 0`
 *  - `tensor([4]) = [0, 0, 0, 0]`
 *  - `tensor([2, 3]) = [[0, 0, 0], [0, 0, 0]]`
 *  - `tensor([4, 7, 6]) = [...]`
 * 
 * @param {number[]} dimensions 
 */
function tensor(dimensions) {
    if (!dimensions.length)
        return 0;

    const result = new Array(dimensions[0]);

    for (let i = 0; i < result.length; i++)
        result[i] = tensor(dimensions.slice(1));

    return result;
}