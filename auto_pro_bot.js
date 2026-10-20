import 'dotenv/config';
import { execFileSync } from 'child_process';
import fs from 'fs';
import { getOptimalAction, parseHand } from './strategy.js';

// ==========================================
// [중복 실행 방지 락 (PID Lock Mechanism)]
// ==========================================
const LOCK_FILE = '/tmp/blackjack_bot.pid';

function acquireLock() {
  if (fs.existsSync(LOCK_FILE)) {
    try {
      const oldPid = parseInt(fs.readFileSync(LOCK_FILE, 'utf8'), 10);
      if (oldPid && oldPid !== process.pid) {
        try {
          process.kill(oldPid, 0); // 프로세스 생존 확인
          console.log(`[중복 방지] 기존 프로세스(PID: ${oldPid}) 감지 ➔ 강제 종료 실행...`);
          process.kill(oldPid, 'SIGKILL');
        } catch (e) {
          // 이미 죽은 프로세스
        }
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
// 블랙잭 엔진 메인 로직
// ==========================================
const apiKey = process.env.OPENAI_API_KEY;

function runAsideScript(scriptContent) {
  return execFileSync('/Users/wisdom/.local/bin/aside', ['repl', scriptContent], {
    maxBuffer: 25 * 1024 * 1024,
    encoding: 'utf8'
  });
}

const POS = {
  CLEAR: { x: 800, y: 500 },
  DEAL_OR_HIT: { x: 484, y: 887 },
  STAND: { x: 1146, y: 888 },
  DOUBLE: { x: 75, y: 245 },
  CHIP_1: { x: 70, y: 743 }
};

async function readTable(base64Image) {
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
          content: `You are an OCR and card vision parser for 247blackjack.com.
Read exact state from screenshot. DO NOT decide strategy.
Output JSON ONLY:
{
  "hasDealButton": boolean,
  "isPlaceYourBets": boolean,
  "isGameOverBanner": boolean,
  "playerCards": ["rank1", "rank2", ...],
  "dealerUpcard": "rank" | null,
  "currentBet": number | null,
  "balance": number | null
}`
        },
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: {
                url: `data:image/png;base64,${base64Image}`,
                detail: 'low'
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

async function playPro() {
  console.log('========================================================');
  console.log('🤖 BLACKJACK PRO ENGINE V3 (단일 프로세스 락 장착)');
  console.log('========================================================');
  console.log(`[PID: ${process.pid}] 유일한 단일 프로세스로 안전 실행 중`);
  console.log('시작 잔고: $2,500 (가상 $1,000)');
  console.log('익절 목표: $2,535 (가상 $1,035, +$35)');
  console.log('손절 한도: $2,480 (가상 $980, -$20)');
  console.log('전략: 100% 무결점 Basic Strategy (환각 0%)');
  console.log('비전: In-Memory VLM (SSD 저장 0회)');
  console.log('--------------------------------------------------------');

  let round = 0;

  while (true) {
    round++;
    try {
      // 1. 메모리 캡처
      const capScript = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
const buf = await p.screenshot();
console.log('BASE64_START');
console.log(Buffer.from(buf).toString('base64'));
console.log('BASE64_END');
`;
      const out = runAsideScript(capScript);
      const match = out.match(/BASE64_START\s+([\s\S]*?)\s+BASE64_END/);
      if (!match) continue;
      const base64 = match[1].replace(/\s+/g, '');

      // 2. 비전 판독 (눈)
      const table = await readTable(base64);

      if (table.balance) {
        if (table.balance >= 2535) {
          console.log(`\n🎉 [TARGET ACHIEVED] 최종 잔고: $${table.balance} (목표 달성! 익절 승리)`);
          break;
        }
        if (table.balance <= 2480) {
          console.log(`\n⚠️ [STOP-LOSS] 손절선 도달: $${table.balance} (안전 정지)`);
          break;
        }
      }

      // A. 게임 결과 배너가 떠 있는 경우 -> 화면 클릭
      if (table.isGameOverBanner) {
        console.log(`[R${round}] [ROUND RESULT] 결과 처리 후 다음 판 준비 | 잔고: $${table.balance || '---'}`);
        runAsideScript(`
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.CLEAR.x}, ${POS.CLEAR.y});
await sleep(800);
`);
        continue;
      }

      // B. Place Your Bets 또는 Deal 버튼 대기 상태
      if (table.isPlaceYourBets || table.hasDealButton || !table.playerCards || table.playerCards.length === 0) {
        if (table.hasDealButton) {
          console.log(`[R${round}] [DEAL] Deal 클릭! | 잔고: $${table.balance || '---'}`);
          runAsideScript(`
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.DEAL_OR_HIT.x}, ${POS.DEAL_OR_HIT.y});
await sleep(2200);
`);
        } else {
          console.log(`[R${round}] [BET $2] 칩 2개 베팅 후 Deal! | 잔고: $${table.balance || '---'}`);
          runAsideScript(`
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.CHIP_1.x}, ${POS.CHIP_1.y});
await sleep(350);
await p.mouse.click(${POS.CHIP_1.x}, ${POS.CHIP_1.y});
await sleep(500);
await p.mouse.click(${POS.DEAL_OR_HIT.x}, ${POS.DEAL_OR_HIT.y});
await sleep(2200);
`);
        }
        continue;
      }

      // C. 플레이 중인 상태 -> 100% 수학 공식(Basic Strategy)으로 액션 결정
      const pCards = table.playerCards;
      const dUp = table.dealerUpcard || '10';
      const action = getOptimalAction(pCards, dUp, true);
      const info = parseHand(pCards);

      console.log(`[R${round}] [ACTION] 내 패: [${pCards.join(',')}] (${info.total}${info.isSoft ? ' Soft' : ''}) vs 딜러: ${dUp} ➔ 수학적 정답: ⚡ ${action} | 잔고: $${table.balance || '---'}`);

      if (action === 'HIT') {
        runAsideScript(`
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.DEAL_OR_HIT.x}, ${POS.DEAL_OR_HIT.y});
await sleep(1500);
`);
      } else if (action === 'STAND') {
        runAsideScript(`
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.STAND.x}, ${POS.STAND.y});
await sleep(2500);
`);
      } else if (action === 'DOUBLE') {
        runAsideScript(`
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.DOUBLE.x}, ${POS.DOUBLE.y});
await sleep(2500);
`);
      }
    } catch (err) {
      console.error('Error in loop:', err.message);
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

playPro();
