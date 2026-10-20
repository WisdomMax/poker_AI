import { execFileSync } from 'child_process';
import fs from 'fs';

export function runAsideScript(scriptContent) {
  const output = execFileSync('/Users/wisdom/.local/bin/aside', ['repl', scriptContent], {
    maxBuffer: 25 * 1024 * 1024,
    encoding: 'utf8'
  });
  return output;
}

export function captureAndSave(filename = 'current_table.png') {
  const script = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
const buf = await p.screenshot();
console.log('BASE64_START');
console.log(Buffer.from(buf).toString('base64'));
console.log('BASE64_END');
`;
  const out = runAsideScript(script);
  const match = out.match(/BASE64_START\s+([\s\S]*?)\s+BASE64_END/);
  if (!match) throw new Error('스크린샷 캡처 실패');
  const base64 = match[1].replace(/\s+/g, '');
  fs.writeFileSync(filename, Buffer.from(base64, 'base64'));
  return filename;
}

export function clickCanvasRelative(x, y) {
  const script = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
const frame = p.frames().find(f => f.url().includes('frame.html') || f.name() === 'game');
const canvas = await frame.$('canvas');
await canvas.click({ position: { x: ${x}, y: ${y} } });
await sleep(1500);
console.log('CLICK_SUCCESS');
`;
  return runAsideScript(script);
}

// 직접 실행 테스트
if (process.argv[1] && process.argv[1].endsWith('bot.js')) {
  console.log('Attempting Deal button click...');
  const out = clickCanvasRelative(484, 816);
  console.log(out);
  captureAndSave();
  console.log('Done, captured table.');
}
