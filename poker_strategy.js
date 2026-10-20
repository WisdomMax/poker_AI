/**
 * 텍사스 홀덤 정석 전략 엔진 (Texas Hold'em GTO & Tight-Aggressive Strategy)
 */

const RANK_VALUE = {
  '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
  '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
};

export function evaluatePreflop(card1, card2) {
  const r1 = RANK_VALUE[card1.toUpperCase()] || 0;
  const r2 = RANK_VALUE[card2.toUpperCase()] || 0;
  const high = Math.max(r1, r2);
  const low = Math.min(r1, r2);
  const isPair = r1 === r2;

  // 1. 프리미엄 핸드 (AA, KK, QQ, JJ, TT, AK)
  if (isPair && high >= 10) return 'RAISE';
  if (high === 14 && low === 13) return 'RAISE'; // AK

  // 2. 플레이 가능한 강한 핸드 (99~66, AQ, AJ, KQ, KJ)
  if (isPair && high >= 6) return 'CALL';
  if (high === 14 && low >= 10) return 'CALL'; // AQ, AJ, AT
  if (high === 13 && low >= 11) return 'CALL'; // KQ, KJ
  if (high === 12 && low === 11) return 'CALL'; // QJ

  // 3. 약한 핸드 -> 무조건 폴드 (칩 아끼기)
  return 'FOLD';
}

/**
 * 포스트플랍 (플랍/턴/리버) 액션 평가
 * @param {Array<string>} myCards 내 카드 2장
 * @param {Array<string>} boardCards 커뮤니티 카드
 * @param {boolean} canCheck 공짜 Check 가능 여부
 */
export function evaluatePostflop(myCards, boardCards, canCheck = false) {
  const allCards = [...myCards, ...boardCards].map(c => RANK_VALUE[c.toUpperCase()] || 0);
  const myRanks = myCards.map(c => RANK_VALUE[c.toUpperCase()] || 0);

  // 빈도수 계산
  const counts = {};
  for (const r of allCards) {
    counts[r] = (counts[r] || 0) + 1;
  }

  const pairs = Object.entries(counts).filter(([_, count]) => count === 2).map(([r]) => Number(r));
  const trips = Object.entries(counts).filter(([_, count]) => count === 3).map(([r]) => Number(r));
  const quads = Object.entries(counts).filter(([_, count]) => count === 4).map(([r]) => Number(r));

  const hasMadeHand = myRanks.some(r => counts[r] >= 2);

  // 1. 몬스터 핸드 (풀하우스, 트립스, 투페어 이상)
  if (quads.length > 0 || (trips.length > 0 && pairs.length > 0) || trips.length > 0) {
    return 'RAISE';
  }
  if (pairs.length >= 2 && myRanks.some(r => pairs.includes(r))) {
    return 'RAISE'; // 투페어
  }

  // 2. 원 페어 (탑 페어 이상)
  if (hasMadeHand) {
    const highestBoard = Math.max(...boardCards.map(c => RANK_VALUE[c.toUpperCase()] || 0));
    const myPair = myRanks.find(r => counts[r] === 2);
    if (myPair && myPair >= highestBoard) {
      return 'RAISE'; // 탑 페어
    }
    return 'CALL'; // 미들/바텀 페어
  }

  // 3. 아무것도 안 맞은 경우 (No Made Hand)
  if (canCheck) return 'CALL'; // Call 버튼 위치가 Check일 때는 공짜 카드 봄
  return 'FOLD'; // 상대가 벳했을 때 노페어면 100% 폴드
}
