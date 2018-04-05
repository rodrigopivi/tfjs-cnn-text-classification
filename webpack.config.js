const { resolve } = require('path');

const config = {
  entry: ['babel-polyfill', './src/index.ts'],
  mode: 'development',
  output: {
    filename: 'index.js',
    path: resolve(__dirname, 'public')
  },
  devServer: {
    contentBase: resolve(__dirname, "public"),
    compress: true,
    port: 3000
  },
  module: {
    rules: [
      {
        test: /\.(js|ts)$/,
        exclude: [/\/node_modules\//],
        use: 'babel-loader'
      },
    ]
  }
}

module.exports = config