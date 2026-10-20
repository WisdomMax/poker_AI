import 'dotenv/config';
import { execFileSync } from 'child_process';
import fs from 'fs';
import { evaluatePreflop, evaluatePostflop } from './poker_strategy.js';

// ==========================================
// [중복 실행 방지 락 (PID Lock)]
// ==========================================
const LOCK_FILE = '/tmp/poker_bot.pid';

function acquireLock() {
  if (fs.existsSync(LOCK_FILE)) {
    try {
      const oldPid = parseInt(fs.readFileSync(LOCK_FILE, 'utf8'), 10);
      if (oldPid && oldPid !== process.pid) {
        try {
          process.kill(oldPid, 0);
          console.log(`[중복 방지] 기존 포커 프로세스(PID: ${oldPid}) 감지 ➔ 강제 종료 실행...`);
          process.kill(oldPid, 'SIGKILL');
        } catch (e) {}
      }
    } catch (e) {}
  }
  fs.writeFileSync(LOCK_FILE, String(process.pid));
}

function releaseLock() {
  try {
    if (fs.existsSync(LOCK_FILE)) fs.unlinkSync(LOCK_FILE);
  } catch (e) {}
}

process.on('exit', releaseLock);
process.on('SIGINT', () => { releaseLock(); process.exit(); });
process.on('SIGTERM', () => { releaseLock(); process.exit(); });

acquireLock();

// ==========================================
// 포커 봇 메인 루프
// ==========================================
const apiKey = process.env.OPENAI_API_KEY;
const POKER_TAB_ID = 'E5CA5A9A10C565E6C34B3FD3A70E46C1';

function runAsideScript(scriptContent) {
  return execFileSync('/Users/wisdom/.local/bin/aside', ['repl', scriptContent], {
    maxBuffer: 25 * 1024 * 1024,
    encoding: 'utf8'
  });
}

const POS = {
  FOLD: { x: 407, y: 1148 },
  CALL_CHECK: { x: 723, y: 1148 },
  RAISE: { x: 1036, y: 1148 },
  SLIDER_OK: { x: 1385, y: 1148 },
  SLIDER_CANCEL: { x: 50, y: 1148 },
  CENTER: { x: 800, y: 500 }
};

// VLM은 카드와 테이블 상태만 읽는 눈 역할 (high detail)
async function readPokerTable(base64Image) {
  const resp = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are an OCR and vision parser for 247freepoker.com Texas Hold'em.
Read exact state from screenshot. DO NOT decide strategy.
Output JSON ONLY:
{
  "isDealAgain": boolean, // true if 'Deal Again' or 'Press anywhere to continue' is visible
  "isRaiseSlider": boolean, // true if red horizontal slider bar with OK button is visible at bottom
  "hasActionButtons": boolean, // true if bottom has Fold, Call/Check, Raise buttons
  "callButtonText": "Call" | "Check" | null,
  "myCards": ["rank1", "rank2"] | [], // player's hole cards near bottom-center 'You' label, e.g. ["3", "10"] or ["A", "K"]
  "boardCards": ["rank", ...] | [], // community cards in middle of table
  "pot": number | null, // number near $ at top/middle
  "myBalance": number | null // CRITICAL: Look ONLY at the red box directly above the yellow 'You' label at bottom-center! If it says $1,000 return 1000. DO NOT read Einstein, Grace, Ada, or Hedy!
}`
        },
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: {
                url: `data:image/png;base64,${base64Image}`,
                detail: 'high'
              }
            }
          ]
        }
      ],
      response_format: { type: 'json_object' },
      temperature: 0
    })
  });

  const data = await resp.json();
  return JSON.parse(data.choices[0].message.content);
}

async function playPoker() {
  console.log('========================================================');
  console.log('🃏 TEXAS HOLD\'EM AI BOT (수학 전략 + VLM 눈 파이프라인)');
  console.log('========================================================');
  console.log(`[PID: ${process.pid}] 포커 단일 프로세스 정상 가동`);
  console.log('시작 잔고: $1,000');
  console.log('익절 목표: $1,035 (+$35) | 손절 한도: $900 (-$100)');
  console.log('전략: Tight-Aggressive (TAG) & 슬라이더 UI 무결점 방어');
  console.log('--------------------------------------------------------');

  let round = 0;

  while (true) {
    round++;
    try {
      // 1. 메모리 캡처: 전체 테이블 + 내 잔고 핀포인트 클립
      const capScript = `
const p = await attachBrowserTab('${POKER_TAB_ID}');
const buf = await p.screenshot();
const balBuf = await p.screenshot({ clip: { x: 550, y: 800, width: 350, height: 250 } });
console.log('BASE64_START');
console.log(Buffer.from(buf).toString('base64'));
console.log('BASE64_END');
console.log('BAL_START');
console.log(Buffer.from(balBuf).toString('base64'));
console.log('BAL_END');
`;
      const out = runAsideScript(capScript);
      const match = out.match(/BASE64_START\s+([\s\S]*?)\s+BASE64_END/);
      const balMatch = out.match(/BAL_START\s+([\s\S]*?)\s+BAL_END/);
      if (!match) continue;
      const base64 = match[1].replace(/\s+/g, '');
      const balBase64 = balMatch ? balMatch[1].replace(/\s+/g, '') : null;

      // 2. 비전 분석
      const table = await readPokerTable(base64);
      
      // 내 잔고는 핀포인트 클립에서 전용으로 파싱 (다른 플레이어 시야 원천 차단)
      let currentBalance = table.myBalance;
      if (balBase64) {
        try {
          const balResp = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: 'gpt-4o-mini',
              messages: [{ role: 'user', content: [{ type: 'text', text: 'Read the number next to $ inside the red rectangle (e.g. 997, 1000). Return JSON: {"balance": number}' }, { type: 'image_url', image_url: { url: `data:image/png;base64,${balBase64}` } }] }],
              response_format: { type: 'json_object' },
              temperature: 0
            })
          });
          const balData = await balResp.json();
          const parsed = JSON.parse(balData.choices[0].message.content).balance;
          if (parsed && typeof parsed === 'number') currentBalance = parsed;
        } catch(e) {}
      }

      if (currentBalance !== null && currentBalance !== undefined) {
        if (currentBalance >= 1035) {
          console.log(`\n🎉 [TARGET ACHIEVED] 최종 잔고: $${currentBalance} (포커 목표 달성! 익절 승리)`);
          break;
        }
        if (currentBalance <= 900) {
          console.log(`\n⚠️ [STOP-LOSS] 손절 한도 도달: $${currentBalance} (안전 정지)`);
          break;
        }
      }

      // A. 핸드 종료 및 Deal Again 화면 -> 화면 중앙 클릭하여 새 판 시작
      if (table.isDealAgain) {
        console.log(`[R${round}] [DEAL AGAIN] 새 핸드 시작 클릭 | 잔고: $${currentBalance || '---'}`);
        runAsideScript(`
const p = await attachBrowserTab('${POKER_TAB_ID}');
await p.bringToFront();
await p.mouse.click(${POS.CENTER.x}, ${POS.CENTER.y});
await sleep(2500);
`);
        continue;
      }

      // B. 레이즈 슬라이더 화면이 뜬 경우 -> OK를 눌러 확정 진행
      if (table.isRaiseSlider) {
        console.log(`[R${round}] [SLIDER UI 감지] ➔ OK 버튼 클릭으로 베팅 확정 | 잔고: $${currentBalance || '---'}`);
        runAsideScript(`
const p = await attachBrowserTab('${POKER_TAB_ID}');
await p.bringToFront();
await p.mouse.move(${POS.SLIDER_OK.x}, ${POS.SLIDER_OK.y});
await sleep(100);
await p.mouse.down();
await sleep(150);
await p.mouse.up();
await sleep(2500);
`);
        continue;
      }

      // C. 내 턴이 와서 버튼이 활성화된 상태
      if (table.hasActionButtons && table.myCards && table.myCards.length >= 2) {
        let action = 'FOLD';
        const isPreflop = !table.boardCards || table.boardCards.length === 0;
        const canCheck = table.callButtonText === 'Check';

        if (isPreflop) {
          action = evaluatePreflop(table.myCards[0], table.myCards[1]);
          if (action === 'FOLD' && canCheck) action = 'CALL'; // 체크가 공짜면 폴드 대신 체크
          console.log(`[R${round}] [PREFLOP] 내 패: [${table.myCards.join(',')}] ➔ 수학 정석: ⚡ ${action} | 팟: $${table.pot || '---'} | 잔고: $${currentBalance || '---'}`);
        } else {
          action = evaluatePostflop(table.myCards, table.boardCards, canCheck);
          console.log(`[R${round}] [POSTFLOP] 내 패: [${table.myCards.join(',')}] | 보드: [${table.boardCards.join(',')}] ➔ 전략: ⚡ ${action} | 팟: $${table.pot || '---'} | 잔고: $${currentBalance || '---'}`);
        }

        // 클릭 집행
        let clickPos = POS.FOLD;
        if (action === 'CALL') clickPos = POS.CALL_CHECK;
        if (action === 'RAISE') clickPos = POS.RAISE;

        if (action === 'RAISE') {
          // 레이즈 클릭 후 나타나는 슬라이더의 OK 버튼까지 2단계 연속 클릭
          runAsideScript(`
const p = await attachBrowserTab('${POKER_TAB_ID}');
await p.bringToFront();
await p.mouse.move(${POS.RAISE.x}, ${POS.RAISE.y});
await sleep(100);
await p.mouse.down();
await sleep(150);
await p.mouse.up();
await sleep(800);
await p.mouse.move(${POS.SLIDER_OK.x}, ${POS.SLIDER_OK.y});
await sleep(100);
await p.mouse.down();
await sleep(150);
await p.mouse.up();
await sleep(3000);
`);
        } else {
          runAsideScript(`
const p = await attachBrowserTab('${POKER_TAB_ID}');
await p.bringToFront();
await p.mouse.move(${clickPos.x}, ${clickPos.y});
await sleep(100);
await p.mouse.down();
await sleep(150);
await p.mouse.up();
await sleep(3000);
`);
        }
        continue;
      }

      // D. 상대방들이 플레이 중인 대기 상태
      await new Promise(r => setTimeout(r, 1000));
    } catch (err) {
      console.error('Poker Loop Error:', err.message);
      await new Promise(r => setTimeout(r, 1500));
    }
  }
}

playPoker();
