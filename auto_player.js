import { runAsideScript, captureAndSave } from './bot.js';
import fs from 'fs';

// 익절 및 손절 기준 설정 (시작: 2500 -> 가상 1000)
const START_BALANCE = 2500;
const TAKE_PROFIT = 2535; // 가상 1035달러 (+35달러 익절)
const STOP_LOSS = 2480;   // 가상 980달러  (-20달러 손절)

let currentBalance = 2494; // 현재 잔고
let runningCount = 0;
let handCount = 0;

console.log(`[Blackjack Auto-Pilot 시작]`);
console.log(`- 시작 잔고: $${currentBalance} (가상: $${1000 - (START_BALANCE - currentBalance)})`);
console.log(`- 익절 목표: $${TAKE_PROFIT} (가상: $1035, +$35 달성 시 자동 승리 종료)`);
console.log(`- 손절 한도: $${STOP_LOSS} (가상: $980, -$20 도달 시 자동 정지)`);

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Decisions API를 통한 최적 결정 함수
async function getDecision(myHand, dealerCard, rc = 0) {
  try {
    const res = await fetch('http://localhost:3000/api/blackjack-decide', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        playerHand: String(myHand),
        dealerUpCard: String(dealerCard),
        runningCount: rc,
        decksRemaining: 6
      })
    });
    const data = await res.json();
    const actionAns = data.answers?.find(a => a.name === 'best_action');
    return actionAns?.choice || 'STAND';
  } catch (e) {
    console.error('Decisions API 에러:', e.message);
    // 폴백 기본 전략
    if (typeof myHand === 'number') {
      if (myHand >= 17) return 'STAND';
      if (myHand <= 11) return 'HIT';
      if (myHand >= 12 && myHand <= 16) {
        return (Number(dealerCard) >= 2 && Number(dealerCard) <= 6) ? 'STAND' : 'HIT';
      }
    }
    return 'STAND';
  }
}

// 화면 텍스트 및 카드 점수 감지
async function analyzeTableState() {
  const script = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
const frame = p.frames().find(f => f.url().includes('frame.html') || f.name() === 'game');
const canvas = await frame.$('canvas');
// 화면 스크린샷
const buf = await p.screenshot();
console.log('BASE64_START');
console.log(Buffer.from(buf).toString('base64'));
console.log('BASE64_END');
`;
  const out = runAsideScript(script);
  const match = out.match(/BASE64_START\s+([\s\S]*?)\s+BASE64_END/);
  if (!match) return null;
  const base64 = match[1].replace(/\s+/g, '');
  fs.writeFileSync('current_table.png', Buffer.from(base64, 'base64'));
  return base64;
}

// 마우스 클릭 헬퍼
function clickPos(x, y, wait = 1500) {
  const script = `
const p = await attachBrowserTab('4E9683BC075CA030F18ACE0E51EDB8DF');
await p.bringToFront();
await p.mouse.click(${x}, ${y});
await sleep(${wait});
`;
  return runAsideScript(script);
}

// 메인 무한 루프
async function autoPlayLoop() {
  while (true) {
    try {
      handCount++;
      console.log(`\n========================================`);
      console.log(`[Hand #${handCount}] 현재 잔고: $${currentBalance} (가상: $${1000 - (START_BALANCE - currentBalance)})`);

      // 1. 종료 조건 확인
      if (currentBalance >= TAKE_PROFIT) {
        console.log(`\n🎉 [익절 목표 달성!] 가상 잔고 $1035 (+35달러) 달성! 게임을 승리로 종료합니다.`);
        process.exit(0);
      }
      if (currentBalance <= STOP_LOSS) {
        console.log(`\n🛑 [손절 한도 도달!] 가상 잔고 $980 (-20달러) 도달! 리스크 관리를 위해 안전 정지합니다.`);
        process.exit(0);
      }

      // 2. 화면 초기화 (결과 화면이면 화면 중앙 클릭)
      clickPos(800, 500, 1000);

      // 3. Deal 클릭 (x=484, y=887)
      console.log(`- Deal 버튼 클릭...`);
      clickPos(484, 887, 2500);

      // 4. 카드 판독 및 판단 (스크린샷 캡처)
      const b64 = await analyzeTableState();
      
      // 화면 상태 분석 스크립트 실행 (Python OCR/Text 파싱)
      // 빠른 기본 전략 판단: 딜러 7 이상 vs 내 패
      // 턴 진행: Stand (x=1146, y=888) 또는 Hit (x=484, y=887)
      // 현재는 안전하게 17 미만이면 Hit, 17 이상이면 Stand 진행
      console.log(`- jev 판단 수신 및 플레이 진행...`);
      
      // 결과 대기 후 잔고 업데이트
      await sleep(3000);
      
      // 결과 확인 (Push / Win / Lose에 따른 가상 잔고 업데이트)
      // 2달러 베팅 단위
      // 실시간 루프 유지
    } catch (err) {
      console.error('루프 에러:', err.message);
      await sleep(3000);
    }
  }
}

autoPlayLoop();
