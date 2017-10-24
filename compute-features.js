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
    { name: 'moves', alias: 'm', type: Number },
    { name: 'pattern', alias: 'p', type: String }
]);

console.log(args);

for (const dir of sgf.dirs) {
    for (const file of dir.problems) {
        if (args.pattern && file.path.indexOf(args.pattern) < 0)
            continue;

        console.log('\n[-] ' + file.path);
        const sgf = file + '';

        try {
            // not all SGF files contain annotated problems
            const sgfp = tsumego.SGF.parse(sgf);
            /* const plays = tsumego.stone.label.color(sgfp.steps[0].PL[0]);

            if (!plays || !sgfp.vars.length) {
                console.log('[!] this is not an annotated problems');
                continue;
            } */

            const solver = new tsumego.Solver(sgf);
            const board = solver.board;
            const color = tsumego.sign(board.get(solver.target));

            console.log('[?] solving the problem...');
            const move = solver.solve(-color, -color);
            const safe = tsumego.stone.color(move) * -color > 0 ? 0 : 1;

            console.log('[?] generating subproblems...');
            const tree = maketree(sgf);
            //console.log(tree);

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
            console.log('[!]', err);
        }
    }
}

/**
 * Gets the initial position and generates a tree
 * of moves that alter the status of the target group,
 * i.e. if in the initial position the group is safe,
 * this function will find moves that make the group
 * unsafe, then it will find moves that make it back
 * safe and so on.
 * 
 * The output is a tree of moves in the SGF format.
 * 
 * @param {string} sgf
 * @returns {string}
 */
function maketree(sgf) {
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));

    function isTargetSafe() {
        const move = solver.solve(-color, -color);
        return tsumego.stone.color(move) * color > 0;
    }

    function isTargetCaptured() {
        return !board.get(solver.target);
    }

    function expand(root, depth, safe) {
        // Use DFS to find the leaf nodes at the given depth.
        // This could be avoided if leaf nodes stored the entire
        // board state which would be possible if the solver could
        // switch to an arbitrary position on demand. This in turn
        // is possible as long as the new board has the same hashes:
        // otherwise the solver's cache won't be useful.
        if (depth > 0) {
            let count = 0;

            for (const move in root) {
                //console.log('adding ' + move);
                board.play(tsumego.stone.fromString(move));
                count += expand(root[move], depth - 1, !safe);
                board.undo();
            }

            return count;
        }

        if (isTargetCaptured())
            return 0;

        const moves = [...solver.getValidMovesFor(safe ? -color : color)];

        // skip trivial positions with only a few possible moves
        if (moves.length < args.moves)
            return 0;

        let count = 0;

        // Now find moves that change the status of the target:
        // if it's safe, find moves that make it unsafe;
        // if it's unsafe, find moves that make it safe.
        for (const move of moves) {
            //console.log('trying ' + tsumego.stone.toString(move));
            board.play(move);

            if (isTargetSafe() != safe) {
                //console.log('this move changes status');
                root[tsumego.stone.toString(move)] = {};
                count++;
            }

            board.undo();
        }

        return count;
    }

    // stringify({}) == ""
    function stringify(root, depth) {
        const variations = [];

        for (const move in root) {
            const subtree = stringify(root[move], depth + 1);
            variations.push(';' + move + subtree);
        }

        return variations.length > 1 ?
            variations.map(s => '\n' + ' '.repeat(depth * 4) + '(' + s + ')').join('') :
            variations.join('');
    }

    const tree = {}; // tree["B[fi]"] = subtree
    const safe = isTargetSafe();

    for (let depth = 0; depth < args.tree; depth++) {
        //console.log('expanding depth ' + depth);
        const count = expand(tree, depth, safe);
        console.log('added ' + count + ' new positions at depth ' + depth);
        //console.log(tree);
        if (!count) break;
    }

    return stringify(tree, 0);
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