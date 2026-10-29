const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('MathJax waits for startup and serializes repeated navigation', async () => {
  let emit, ready, active = 0, maximum = 0, typesets = 0;
  const calls = [];
  const context = {
    window: {}, console,
    document$: {subscribe(callback) {emit = callback; callback();}},
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../docs/javascripts/mathjax.js'), 'utf8'), context);
  const math = context.MathJax = context.window.MathJax;
  math.startup.defaultReady = () => {
    math.startup.promise = new Promise(resolve => {ready = resolve;});
  };
  math.startup.output = {clearCache() {calls.push('cache');}};
  math.typesetClear = () => calls.push('clear');
  math.texReset = () => calls.push('reset');
  math.typesetPromise = async () => {
    maximum = Math.max(maximum, ++active);
    typesets++;
    calls.push('typeset');
    await new Promise(resolve => setImmediate(resolve));
    active--;
  };
  math.startup.ready();
  emit(); emit();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(typesets, 0, 'no typesetting before startup completes');
  ready();
  for (let i = 0; i < 10; i++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(typesets, 3);
  assert.equal(maximum, 1, 'navigation typesets must never overlap');
  assert.deepEqual(calls, Array(3).fill(['cache', 'clear', 'reset', 'typeset']).flat());
});

test('the documentation does not load the polyfill.io service', () => {
  const config = fs.readFileSync(path.join(__dirname, '../mkdocs.yml'), 'utf8');
  assert.ok(!config.includes('polyfill.io'));
});
