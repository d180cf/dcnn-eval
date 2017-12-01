/**
 * Input: SGF file with a tsumego.
 * Output: SGF file with a tree of moves.
 * 
 * The algorithm that generates the tree of moves is quite simple:
 * 
 *  1. If the status of the board depends on who plays first,
 *     pick a few moves for both players and continue with them.
 * 
 *  2. If the target group is safe, i.e. it cannot be captured,
 *     pick a few moves that try to attack it.
 * 
 *  3. If the target group is dead, i.e. it cannot be saved,
 *     pick a few moves that try to defend it.
 * 
 * The number of moves picked at each iteration is the branching factor.
 * They can be picked randomly or they can be picked by the policy DNN.
 */

const tsumego = require('tsumego.js');
const fstext = require('./fstext');

const [, , input, output, maxTreeSize, minAreaSize, branchingFactor, verbose] = process.argv;

try {
    const sgf = fstext.read(input, 'utf8');
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));

    verbose && console.log('solving the problem...');
    verbose && console.log(board.text);
    const move = solver.solve(-color, -color);
    const safe = tsumego.stone.color(move) * -color > 0 ? 0 : 1;

    verbose && console.log('generating subproblems...');
    const tree = maketree(sgf);

    const data = board.sgf.slice(0, -1)
        + 'MA' + tsumego.stone.toString(solver.target)
        + '\n' + tree + ')';

    fstext.write(output, data);
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
    const komaster = -color;
    const nodes = {}; // nodes[board.hash] = true

    let count = 0;
    let count_d = 0;
    let count_s = 0;

    function bestmove(player) {
        return solver.solve(player, komaster);
    }

    // -1 = the target is dead even if it plays first
    // +1 = the target is safe even if it passes
    //  0 = the status depends on who plays first
    function getStatus() {
        const defense = bestmove(+color); // try to save the target
        const attack = bestmove(-color); // try to take the target

        return defense * attack < 0 ? 0 : Math.sign(defense * color);
    }

    function isTargetCaptured() {
        return !board.get(solver.target);
    }

    function mstr(move) {
        return tsumego.stone.toString(move);
    }

    function add(root, move) {
        // even if the opponent is the ko master,
        // there are cases when a ko changes the
        // status of the target group, so it's
        // necessary to remember seen positions        
        if (nodes[board.hash])
            return;

        nodes[board.hash] = true;
        const node = {};
        root[mstr(move)] = node;
        count++;

        // it's non-enumerable to simplify things in other places
        Object.defineProperty(node, 'status', { value: getStatus() });

        if (node.status < 0) count_d++;
        if (node.status > 0) count_s++;
    }

    function expand(root, depth) {
        // Use DFS to find the leaf nodes at the given depth.
        // This could be avoided if leaf nodes stored the entire
        // board state which would be possible if the solver could
        // switch to an arbitrary position on demand. This in turn
        // is possible as long as the new board has the same hashes:
        // otherwise the solver's cache won't be useful.
        if (depth > 0) {
            for (const move in root) {
                if (!board.play(tsumego.stone.fromString(move))) {
                    verbose && console.log('Invalid move: ' + move);
                    continue;
                }

                expand(root[move], depth - 1);
                board.undo();
            }

            return;
        }

        if (isTargetCaptured())
            return;

        const moves1 = [...solver.getValidMovesFor(+color)];
        const moves2 = [...solver.getValidMovesFor(-color)];

        // skip trivial positions with too few possible moves
        if (Math.min(moves1.length, moves2.length) <= minAreaSize)
            return;

        // if the target is dead, pick a few moves that try to defend it;
        // if the target is safe, pick a few moves that try to capture it;
        // if the status depends on who plays first, picks moves from both sides;
        const moves = root.status < 0 ? moves1 :
            root.status > 0 ? moves2 :
                [...moves1, ...moves2];

        // pick the top few strongest moves; well, at this point moves
        // are selected randomly, but they can be ordered by the DNN
        for (let i = 0; i < branchingFactor; i++) {
            const move = moves[Math.random() * moves.length | 0];

            if (!board.play(move)) {
                verbose && console.log('Invalid move: ' + move);
                continue;
            }

            if (!isTargetCaptured())
                add(root, move);

            board.undo();
        }
    }

    // stringify({}) == ""
    function stringify(root, depth) {
        const variations = [];

        for (const move in root) {
            const s = root[move].status;
            const status = s < 0 ? 'TS[-]' : s > 0 ? 'TS[+]' : '';
            const subtree = stringify(root[move], depth + 1);
            variations.push(';' + move + status + subtree);
        }

        return variations.length > 1 || !depth ?
            variations.map(s => '\n' + ' '.repeat(depth * 4) + '(' + s + ')').join('') :
            variations.join('');
    }

    const tree = {}; // tree["B[fi]"] = subtree

    // it's non-enumerable to simplify things in other places
    Object.defineProperty(tree, 'status', { value: getStatus() });

    for (let depth = 0; count < maxTreeSize; depth++) {
        const before = count;
        expand(tree, depth);
        const after = count;
        if (before == after) break;
        verbose && console.log(`depth = ${depth} added ${after - before} nodes`);
    }

    verbose && console.log(`safe = ${count_s} dead = ${count_d}`);
    return stringify(tree, 0);
}
