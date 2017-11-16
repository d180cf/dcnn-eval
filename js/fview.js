/**
 * Input: JSON file with features.
 * Output: text file with their visual representation
 */

const fstext = require('./fstext');

const [, , input, output] = process.argv;

const features = JSON.parse(fstext.read(input));
const text = features.map(m => m.map(a => a.map(x => x ? '#' : '-').join(' ')).join('\n')).join('\n\n');
fstext.write(output, text);
