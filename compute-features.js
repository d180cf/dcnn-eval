/**
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
const clargs = require('command-line-args');
const sgf = require('sgf-problems');
const tsumego = require('tsumego.js');

const args = clargs([
    { name: 'resdir', alias: 'd', type: String, defaultOption: true },
    { name: 'fpsize', alias: 'n', type: Number },
    { name: 'tree', alias: 't', type: Number },
    { name: 'pattern', alias: 'p', type: String }
]);

for (const dir of sgf.dirs) {
    for (const file of dir.problems) {
        if (args.pattern && file.path.indexOf(args.pattern) < 0)
            continue;

        console.log('\n[-] ' + file.path);
        const sgf = file + '';

        try {
            // not all SGF files contain annotated problems
            const sgfp = tsumego.SGF.parse(sgf);
            const plays = tsumego.stone.label.color(sgfp.steps[0].PL[0]);

            if (!plays || !sgfp.vars.length) {
                console.log('[!] this is not an annotated problems');
                continue;
            }

            const solver = new tsumego.Solver(sgf);
            const board = solver.board;
            const color = tsumego.sign(board.get(solver.target));

            console.log('[?] solving the problem...');
            const move = solver.solve(-color, -color);
            const safe = tsumego.stone.color(move) * -color > 0 ? 0 : 1;

            console.log('[?] building the proof tree...');
            const tree = solver.prooftree(plays, -color, args.tree);
            console.log(tree);
            
            console.log('[?] computing the features...');
            const [x, y] = tsumego.stone.coords(solver.target);
            const feat = features(board, { x, y }, args.fpsize);

            const json = JSON.stringify({
                label: safe ? 1 : 0, // safe means the target cannot be captured
                features: feat
            });

            const respath = fspath.join(args.resdir, file.path).replace(/\.sgf$/, '.json');
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