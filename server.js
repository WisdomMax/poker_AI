import 'dotenv/config';
import express from 'express';
const app = express();
const port = Number(process.env.PORT || 3000);
const model = process.env.OPENAI_DECISIONS_MODEL || 'gpt-6-luna';
const apiKey = process.env.OPENAI_API_KEY;

app.use(express.json({ limit: '2mb' }));
app.use(express.static('public'));

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    configured: Boolean(apiKey),
    model,
  });
});

app.post('/api/decide', async (req, res) => {
  const input = typeof req.body?.input === 'string' ? req.body.input.trim() : '';

  if (!input) {
    return res.status(400).json({ error: 'input은 비어 있을 수 없습니다.' });
  }

  if (!apiKey) {
    return res.status(503).json({
      error: 'OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 만들어 키를 넣어주세요.',
    });
  }

  try {
    const response = await fetch('https://api.openai.com/v1/decisions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        input,
        questions: [
        {
          type: 'choice',
          name: 'next_action',
          instructions: '이 게임 상황에서 가장 적절한 다음 행동을 선택하세요.',
          choices: [
            { value: 'continue', description: '현재 전략을 계속 진행한다.' },
            { value: 'defend', description: '안전하게 방어하고 위험을 줄인다.' },
            { value: 'change_strategy', description: '전략을 바꾸고 다른 방법을 시도한다.' },
            { value: 'ask_for_help', description: '도움이나 추가 정보를 요청한다.' },
          ],
        },
        {
          type: 'predicate',
          name: 'dangerous',
          instructions: '지금 행동을 계속하면 큰 손실이나 위험이 발생할 가능성이 높은가?',
        },
        {
          type: 'score',
          name: 'confidence',
          instructions: '이 판단의 확신도를 평가하세요.',
          levels: [
            { label: '낮음', description: '정보가 부족하거나 불확실하다.' },
            { label: '보통', description: '어느 정도 근거가 있다.' },
            { label: '높음', description: '상황과 근거가 명확하다.' },
          ],
        },
      ],
      }),
    });

    const decision = await response.json();
    if (!response.ok) {
      return res.status(response.status).json({
        error: decision?.error?.message || 'OpenAI Decisions API 호출에 실패했습니다.',
      });
    }

    res.json(decision);
  } catch (error) {
    console.error(error);
    res.status(error?.status || 500).json({
      error: error?.message || 'Decisions API 호출에 실패했습니다.',
    });
  }
});

// 블랙잭 전용 Decisions API 엔드포인트
app.post('/api/blackjack-decide', async (req, res) => {
  const { playerHand, dealerUpCard, runningCount = 0, decksRemaining = 6, history = [] } = req.body || {};

  if (!playerHand || !dealerUpCard) {
    return res.status(400).json({ error: 'playerHand(내 패)와 dealerUpCard(딜러 오픈카드)가 필요합니다.' });
  }

  if (!apiKey) {
    return res.status(503).json({ error: 'OPENAI_API_KEY가 설정되지 않았습니다.' });
  }

  const trueCount = decksRemaining > 0 ? (runningCount / decksRemaining).toFixed(1) : runningCount;

  // jev(Decisions API)에게 전달할 정밀 수학적 맥락 구성
  const inputPrompt = `[블랙잭 게임 상황]
- 내 핸드: ${playerHand}
- 딜러 업카드: ${dealerUpCard}
- 러닝 카운트(Running Count): ${runningCount}
- 남은 덱 수(Decks Remaining): ${decksRemaining}덱
- 트루 카운트(True Count): ${trueCount}
- 카드 히스토리: ${history.length ? history.join(', ') : '초기 덱 상태'}
- 기본 룰: 6-Deck Shoe, Dealer stands on 17, Double on any 2 cards, Split allowed.
- 목표: 수학적 기본 전략(Basic Strategy) 및 Hi-Lo 카드 카운팅(Illustrious 18 변형)을 적용해 플레이어 기댓값(EV)이 가장 높은 최적의 결정을 내리세요.`;

  try {
    const response = await fetch('https://api.openai.com/v1/decisions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        input: inputPrompt,
        questions: [
          {
            type: 'choice',
            name: 'best_action',
            instructions: '현재 상황에서 수학적으로 기대값(EV)이 가장 높은 다음 행동을 선택하세요.',
            choices: [
              { value: 'HIT', description: '카드를 한 장 더 받는다.' },
              { value: 'STAND', description: '현재 점수를 유지하고 차례를 마친다.' },
              { value: 'DOUBLE', description: '베팅을 2배로 올리고 카드 단 1장만 더 받는다.' },
              { value: 'SPLIT', description: '두 카드를 분리하여 2개의 손으로 나눈다.' },
              { value: 'SURRENDER', description: '패를 포기하고 베팅의 절반을 회수한다.' }
            ],
          },
          {
            type: 'predicate',
            name: 'player_advantage',
            instructions: '현재 트루 카운트(TC >= +2) 및 딜러 버스트 확률을 고려할 때, 지금 덱 상황이 플레이어에게 통계적으로 유리한 상태인가?',
          },
          {
            type: 'choice',
            name: 'bet_multiplier',
            instructions: '트루 카운트(TC)에 따른 최적의 켈리 베팅 배율(Bet Unit)을 추천하세요.',
            choices: [
              { value: '1x', description: '기본 최소 베팅 (TC <= +1)' },
              { value: '2x', description: '약간 유리함, 베팅 2배 (TC +2)' },
              { value: '3x-4x', description: '상당히 유리함, 베팅 3~4배 (TC +3~+4)' },
              { value: '5x+', description: '압도적 유리함, 최대 베팅 5배 이상 (TC >= +5)' }
            ],
          },
          {
            type: 'score',
            name: 'confidence',
            instructions: '이 판단의 수학적 확신도를 평가하세요.',
            levels: [
              { label: '낮음', description: '경계선 상황 (Basic Strategy와 카운팅 상충)' },
              { label: '보통', description: '안정적인 통계적 판단' },
              { label: '매우 높음', description: '완벽한 수학적 정답 (기본 전략 및 카운팅 일치)' }
            ],
          }
        ],
      }),
    });

    const decision = await response.json();
    if (!response.ok) {
      return res.status(response.status).json({
        error: decision?.error?.message || 'OpenAI Decisions API 호출 실패',
      });
    }

    res.json({
      ...decision,
      calculated: {
        trueCount: Number(trueCount),
        runningCount,
        decksRemaining
      }
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error?.message || '블랙잭 판단 실패' });
  }
});

app.listen(port, () => {
  console.log(`Hangame Decisions demo: http://localhost:${port}`);
});
