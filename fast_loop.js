import { execFileSync } from 'child_process';
import fs from 'fs';

function runAsideScript(scriptContent) {
  return execFileSync('/Users/wisdom/.local/bin/aside', ['repl', scriptContent], {
    maxBuffer: 25 * 1024 * 1024,
    encoding: 'utf8'
  });
}

// 브라우저 탭에 영구 붙어서 고속 루프 실행
const script = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();

// 좌표 정의
const POS = {
  CLEAR: { x: 800, y: 500 },
  DEAL_OR_HIT: { x: 484, y: 887 },
  STAND: { x: 1146, y: 888 },
  CHIP_10: { x: 50, y: 750 }
};

// 1사이클 플레이 함수
console.log('STARTING_FAST_ROUND');
await p.mouse.click(POS.CLEAR.x, POS.CLEAR.y);
await sleep(1000);
await p.mouse.click(POS.DEAL_OR_HIT.x, POS.DEAL_OR_HIT.y);
await sleep(2000);

const buf = await p.screenshot();
console.log('BASE64_START');
console.log(Buffer.from(buf).toString('base64'));
console.log('BASE64_END');
`;

const out = runAsideScript(script);
const match = out.match(/BASE64_START\s+([\s\S]*?)\s+BASE64_END/);
if (match) {
  fs.writeFileSync('fast_current.png', Buffer.from(match[1].replace(/\s+/g, ''), 'base64'));
}
