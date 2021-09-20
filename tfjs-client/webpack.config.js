// Generated using webpack-cli http://github.com/webpack-cli
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
    mode: 'development',
    entry: {
        bundle: './src/index.js',
        worker: './src/worker.js'
    },
    output: {
        path: path.resolve(__dirname, 'dist'),
        globalObject: `(() => {
            if (typeof self !== 'undefined') {
                return self;
            } else if (typeof window !== 'undefined') {
                return window;
            } else if (typeof global !== 'undefined') {
                return global;
            } else {
                return Function('return this')();
            }
        })()`
    },
    devServer: {
        open: true,
        host: 'localhost',
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: 'index.html',
        }),
        new CopyPlugin({
            patterns: [
                { from: 'src/model', to: 'model', force: true },
                { from: 'src/euhhhh.gif', to: '.', force: true },
                { from: 'src/742.mp3', to: '.', force: true }
            ]
        }),
    ],
    module: {
        rules: [
            {
                test: /\.(eot|svg|ttf|woff|woff2|png|jpg|gif)$/,
                type: 'asset',
            },
        ],
    },
    resolve: {
        fallback: {
            "fs": false,
            "util": false
        },
    }
};
