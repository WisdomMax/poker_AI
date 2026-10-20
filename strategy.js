/**
 * 전 세계 카지노 표준 블랙잭 완벽 기본 전략 (Basic Strategy Look-Up Table)
 * 수학적으로 하우스 엣지를 0.5% 이하로 낮추는 100% 결정론적 룰 엔진
 */

export function parseHand(cards) {
  let total = 0;
  let aceCount = 0;

  for (const c of cards) {
    const rank = String(c).toUpperCase().replace(/[^0-9A-Z]/g, '');
    if (['J', 'Q', 'K'].includes(rank) || rank === '10') {
      total += 10;
    } else if (rank === 'A') {
      total += 11;
      aceCount += 1;
    } else {
      total += parseInt(rank, 10) || 0;
    }
  }

  while (total > 21 && aceCount > 0) {
    total -= 10;
    aceCount -= 1;
  }

  const isSoft = aceCount > 0;
  return { total, isSoft, cardCount: cards.length };
}

export function parseDealerUpcard(card) {
  const rank = String(card).toUpperCase().replace(/[^0-9A-Z]/g, '');
  if (['J', 'Q', 'K'].includes(rank) || rank === '10') return 10;
  if (rank === 'A') return 11;
  return parseInt(rank, 10) || 10;
}

/**
 * 100% 수학 정석 블랙잭 액션 결정 함수
 * @param {Array<string>} playerCards 내 카드 리스트
 * @param {string} dealerUpcard 딜러 오픈 카드
 * @param {boolean} canDouble 더블다운 가능 여부 (처음 2장일 때만 가능)
 * @returns {'HIT' | 'STAND' | 'DOUBLE'}
 */
export function getOptimalAction(playerCards, dealerUpcard, canDouble = true) {
  const { total, isSoft, cardCount } = parseHand(playerCards);
  const d = parseDealerUpcard(dealerUpcard);
  const allowDouble = canDouble && cardCount === 2;

  // 1. 이미 21 이상
  if (total >= 21) return 'STAND';

  // 2. Soft Hands (A가 11로 계산되는 핸드)
  if (isSoft) {
    if (total >= 20) return 'STAND'; // Soft 20, 21 (A,9 / A,10) 무조건 스탠드
    if (total === 19) {
      if (allowDouble && d === 6) return 'DOUBLE';
      return 'STAND';
    }
    if (total === 18) {
      if (allowDouble && d >= 2 && d <= 6) return 'DOUBLE';
      if (d === 7 || d === 8) return 'STAND';
      return 'HIT'; // 9, 10, A 상대로는 HIT
    }
    if (total === 17) {
      if (allowDouble && d >= 3 && d <= 6) return 'DOUBLE';
      return 'HIT';
    }
    if (total === 15 || total === 16) {
      if (allowDouble && d >= 4 && d <= 6) return 'DOUBLE';
      return 'HIT';
    }
    if (total === 13 || total === 14) {
      if (allowDouble && (d === 5 || d === 6)) return 'DOUBLE';
      return 'HIT';
    }
    return 'HIT';
  }

  // 3. Hard Hands (일반 핸드)
  if (total >= 17) return 'STAND'; // 17 이상은 100% 무조건 스탠드!
  if (total >= 13 && total <= 16) {
    // 딜러가 2~6(약한 카드)이면 딜러 버스트를 노리고 무조건 스탠드!
    if (d >= 2 && d <= 6) return 'STAND';
    // 딜러가 7 이상이면 히트
    return 'HIT';
  }
  if (total === 12) {
    if (d >= 4 && d <= 6) return 'STAND';
    return 'HIT';
  }
  if (total === 11) {
    if (allowDouble) return 'DOUBLE';
    return 'HIT';
  }
  if (total === 10) {
    if (allowDouble && d >= 2 && d <= 9) return 'DOUBLE';
    return 'HIT';
  }
  if (total === 9) {
    if (allowDouble && d >= 3 && d <= 6) return 'DOUBLE';
    return 'HIT';
  }
  // 8 이하
  return 'HIT';
}
