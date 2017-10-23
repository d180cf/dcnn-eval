/**
 * Usage: node parse-sgf bin/json 2
 * 
 * This script does a bunch of things:
 * 
 *  1. reads SGF files from the sgf-problems repo
 *  2. solves each tsumego
 *  3. picks all relevant subproblems
 *  4. computes all relevant features
 *  5. generates JSON files with those features
 * 
 * Then a Python script reads the JSON files
 * and feeds to TensorFlow. The output of TF
 * is a large NN that evaluates the status of
 * a given tsumego.
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
            const [x, y] = tsumego.stone.coords(solver.target);
            const json = JSON.stringify(features(solver.board, { x, y }, fpsize));
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
 * The features are computed reltive to
 * the `target` stone: it appears in the
 * middle of each feature plane.
 * 
 * Then the returned tensor has shape `[F, N, N]`
 * where `F` is the number of features and 
 * and `N = (2*size + 1)`.
 * 
 * The computed feature are:
 * 
 *  1. ally = same colored stone as target
 *  2. enemy = the opposite colored stone
 *  3. neutral = the location is outside the board
 *  4. atari = the block has only one liberty
 * 
 * More features to be added, e.g. whether the
 * block can be captured in a ladder (aka the lambda-1
 * sequence), whether it can be captured with
 * a net (aka the lambda-2 sequence) and so on.
 * 
 * @param {tsumego.Board} board
 * @param {{x: number, y: number}} target
 * @param {number} size
 * @returns {number[][][]}
 */
function features(board, target, size) {
    const result = tensor([4, size * 2 + 1, size * 2 + 1]);

    const FI_A = 0;
    const FI_E = 1;
    const FI_N = 2;
    const FI_1 = 3;

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
            } else {
                const block = board.get(x, y);
                const nlibs = tsumego.block.libs(block);

                result[FI_A][i][j] = block * targetColor > 0 ? 1 : 0;
                result[FI_E][i][j] = block * targetColor < 0 ? 1 : 0;
                result[FI_N][i][j] = 0;
                result[FI_1][i][j] = nlibs == 1 ? 1 : 0;
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