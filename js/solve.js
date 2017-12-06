/**
 * Input: SGF file with a tsumego.
 * Output: SGF file with a tree of safe-unsafe lines.
 */

const fstext = require('./fstext');
const fspath = require('path');
const mkdirp = require('mkdirp');
const tsumego = require('tsumego.js');

const [, , input, outputDir, maxTreeDepth, minAreaSize] = process.argv;

try {
    const sgf = fstext.read(input);
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));

    mkdirp.sync(outputDir);

    // 1-st tree of moves when the attacker makes the first move
    // 2-nd tree of moves when the defender makes the first move
    for (const player of [-color, +color]) {
        const tree = maketree(sgf, player);

        const data = board.sgf.slice(0, -1)
            + 'PL[' + (player > 0 ? 'B' : 'W') + ']'
            + 'MA' + tsumego.stone.toString(solver.target)
            + '\nC[' + (player * color > 0 ? 'defender' : 'attacker') + ' makes the first move]'
            + '\n' + tree + ')';

        const filename = player * color > 0 ? 'D' : 'A';
        const output = fspath.join(outputDir, filename + '.sgf');

        console.log(output);
        fstext.write(output, data);
    }
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
 * @param {number} player Tells who makes the first move
 * @returns {string}
 */
function maketree(sgf, player) {
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));
    const komaster = -color; // the attacker can recapture any ko
    const cache = {}; // cache[board.hash] = isTargetSafe()

    function isTargetSafe() {
        const move = solver.solve(player, komaster);
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

            if (!isTargetCaptured() && !cache[board.hash]) {
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
