/**
 * Input: SGF files with TR and MA labels.
 * Output: JSON files with target group status.
 */

const fs = require('fs');
const glob = require('glob');
const fspath = require('path');
const mkdirp = require('mkdirp');
const tsumego = require('tsumego.js');

const [, , input, output] = process.argv;

let status0 = 0;
let status1 = 0;

for (const filepath of glob.sync(input)) {
    const data = fs.readFileSync(filepath, 'utf8');
    const sgf = tsumego.SGF.parse(data);
    const board = new tsumego.Board(sgf);

    const MA = sgf.steps[0].MA[0];
    const TR = sgf.steps[0].TR[0];

    const target = tsumego.stone.fromString(MA);
    const lastmv = tsumego.stone.fromString(TR);

    if (!board.get(target) || !board.get(lastmv)) {
        console.log(MA, target, TR, lastmv);
        throw Error('Invalid input: ' + data);
    }

    const status = board.get(target) * board.get(lastmv) > 0 ? 1 : 0;
    const jsonpath = output.replace('*', /(\w+)\.\w+$/.exec(filepath)[1]);
    mkdirp.sync(fspath.dirname(jsonpath));
    fs.writeFileSync(jsonpath, status, 'utf8');

    if (status)
        status1++;
    else
        status0++;
}

console.log('status=0', status0);
console.log('status=1', status1);
