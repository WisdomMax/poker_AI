import 'dotenv/config';
import { execFileSync } from 'child_process';
import fs from 'fs';

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
  CHIP_10: { x: 50, y: 750 } // $10 칩 좌표
};

// Hi-Lo 카운팅 상태
let runningCount = 0;
let totalCardsDealt = 0;
const TOTAL_DECKS = 6;

function updateHiLo(cardRanks) {
  for (const rank of cardRanks) {
    const r = String(rank).toUpperCase();
    if (['2', '3', '4', '5', '6'].includes(r)) {
      runningCount += 1;
    } else if (['10', 'J', 'Q', 'K', 'A'].includes(r)) {
      runningCount -= 1;
    }
    totalCardsDealt++;
  }
}

function getTrueCount() {
  const remainingDecks = Math.max(1, TOTAL_DECKS - (totalCardsDealt / 52));
  return Number((runningCount / remainingDecks).toFixed(1));
}

async function analyzeTable(base64Image) {
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
          content: `You are an expert card-counting Blackjack AI playing on 247blackjack.com.
Inspect the table screenshot.
1. Determine game state:
   - If round finished ('Dealer Wins', 'Push', 'Blackjack', 'Player Wins', or empty table): action 'DEAL'
   - If player's turn to act: 'HIT' or 'STAND' (follow Basic Strategy)
2. Extract all visible card ranks on the table right now (e.g. ["10", "A", "5", "K"]).
3. Extract player's current balance (number near $ at bottom, e.g. 2501).

JSON ONLY:
{
  "action": "HIT" | "STAND" | "DEAL",
  "reason": "short explanation",
  "cards": ["2".."10","J","Q","K","A"],
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

async function playLoop() {
  console.log('=== HI-LO CARD COUNTING & BET SPREAD BOT STARTED ===');
  console.log('목표 잔고: $2,535 (익절) | 손절선: $2,480');
  console.log('베팅 규칙: TC >= +2.0 일 때 $10 칩 베팅, 그 외 기본 $2 베팅');

  let round = 0;
  let lastCardsString = '';

  while (true) {
    round++;
    try {
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

      const res = await analyzeTable(base64);

      // 새 카드 카운팅 반영
      const currentCardsStr = (res.cards || []).sort().join(',');
      if (currentCardsStr && currentCardsStr !== lastCardsString) {
        updateHiLo(res.cards || []);
        lastCardsString = currentCardsStr;
      }

      const tc = getTrueCount();
      const isHighAdvantage = tc >= 2.0;

      console.log(`[R${round}] Action: ${res.action} (${res.reason}) | RC: ${runningCount} | TC: ${tc} ${isHighAdvantage ? '🔥[유리! $10베팅구간]' : ''} | Bal: $${res.balance || '---'}`);

      if (res.balance && res.balance >= 2535) {
        console.log(`🎉 [SUCCESS] 목표 달성! 잔고: $${res.balance}`);
        break;
      }
      if (res.balance && res.balance <= 2480) {
        console.log(`⚠️ [STOP-LOSS] 손절선 도달: $${res.balance}`);
        break;
      }

      let clickScript = '';
      if (res.action === 'DEAL') {
        if (isHighAdvantage) {
          // 유리한 구간: $10 칩 클릭 후 딜!
          clickScript = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.CLEAR.x}, ${POS.CLEAR.y});
await sleep(600);
console.log('Placing $10 High-Advantage Bet...');
await p.mouse.click(${POS.CHIP_10.x}, ${POS.CHIP_10.y});
await sleep(400);
await p.mouse.click(${POS.DEAL_OR_HIT.x}, ${POS.DEAL_OR_HIT.y});
await sleep(1500);
`;
        } else {
          // 기본 구간: 기본 $2 유지 후 딜!
          clickScript = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.CLEAR.x}, ${POS.CLEAR.y});
await sleep(600);
await p.mouse.click(${POS.DEAL_OR_HIT.x}, ${POS.DEAL_OR_HIT.y});
await sleep(1500);
`;
        }
      } else if (res.action === 'HIT') {
        clickScript = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.DEAL_OR_HIT.x}, ${POS.DEAL_OR_HIT.y});
await sleep(1500);
`;
      } else if (res.action === 'STAND') {
        clickScript = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${POS.STAND.x}, ${POS.STAND.y});
await sleep(2000);
`;
      }

      runAsideScript(clickScript);
    } catch (err) {
      console.error('Bot Error:', err.message);
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

playLoop();
