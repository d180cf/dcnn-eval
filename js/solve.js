/**
 * Input: SGF file with a tsumego.
 * Output: SGF file with a tree of safe-unsafe lines.
 */

const fs = require('fs');
const fspath = require('path');
const mkdirp = require('mkdirp');
const tsumego = require('tsumego.js');

const [, , input, output, maxTreeDepth, minAreaSize] = process.argv;

try {
    if (fs.existsSync(output)) {
        console.log('already exists: ' + output);
        process.exit(0);
    }

    const sgf = fs.readFileSync(input, 'utf8');
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));

    console.log('solving the problem...');
    const move = solver.solve(-color, -color);
    const safe = tsumego.stone.color(move) * -color > 0 ? 0 : 1;

    console.log('generating subproblems...');
    const tree = maketree(sgf);

    mkdirp.sync(fspath.dirname(output));
    const data = board.sgf.slice(0, -1)
        + 'MA' + tsumego.stone.toString(solver.target)
        + '\n' + tree + ')';
    fs.writeFileSync(output, data, 'utf8');
} catch (err) {
    throw err;
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
    const cache = {}; // cache[board.hash] = isTargetSafe()

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
        if (moves.length < minAreaSize)
            return 0;

        let count = 0;

        // Now find moves that change the status of the target:
        // if it's safe, find moves that make it unsafe;
        // if it's unsafe, find moves that make it safe.
        for (const move of moves) {
            board.play(move);

            if (!cache[board.hash]) {
                // even if the opponent is the ko master,
                // there are cases when a ko changes the
                // status of the target group, so it's
                // necessary to remember seen positions
                cache[board.hash] = true;

                if (isTargetSafe() != safe) {
                    root[tsumego.stone.toString(move)] = {};
                    count++;
                }
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

        return variations.length > 1 || !depth ?
            variations.map(s => '\n' + ' '.repeat(depth * 4) + '(' + s + ')').join('') :
            variations.join('');
    }

    const tree = {}; // tree["B[fi]"] = subtree
    const safe = isTargetSafe();

    for (let depth = 0; depth < maxTreeDepth; depth++) {
        const count = expand(tree, depth, safe);
        if (!count) break;
    }

    return stringify(tree, 0);
}
