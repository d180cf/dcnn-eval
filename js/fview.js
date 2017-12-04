/**
 * Input: JSON file with features.
 * Output: text file with their visual representation
 */

const fstext = require('./fstext');

const [, , input, output] = process.argv;

const data = JSON.parse(fstext.read(input));
const [size, , depth] = data.shape;

console.log('size = ' + size);
console.log('features = ' + depth);

for (let f = 0; f < depth; f++) {
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let v = data.planes[y * size * depth + x * depth + f];
            let s = v ? '#' : '-';

            process.stdout.write(s + ' ');
        }

        process.stdout.write('\n');
    }

    process.stdout.write('\n');
}
