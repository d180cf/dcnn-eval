const fs = require('fs');
const fspath = require('path');
const mkdirp = require('mkdirp');

/**
 * @param {string} path
 * @returns {string}
 */
exports.read = function read(path) {
    return fs.readFileSync(path, 'utf8');
};

/**
 * @param {string} path
 * @param {string} text
 * @returns {void}
 */
exports.write = function write(path, text) {
    mkdirp.sync(fspath.dirname(path));
    fs.writeFileSync(path, text, 'utf8');
};
