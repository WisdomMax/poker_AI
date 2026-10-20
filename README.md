# Hangame Decisions Demo

OpenAI 공식 Decisions API(`gpt-6-luna`)를 사용하는 작은 로컬 웹 데모입니다.

## 실행

```bash
npm install
cp .env.example .env
# .env의 OPENAI_API_KEY에 OpenAI API 키 입력
npm start
```

브라우저에서 http://localhost:3000 을 엽니다.

## 구조

- `server.js`: API 키를 서버에서만 보관하고 `/api/decide`를 제공
- `public/index.html`: 게임 상황 입력 및 결과 표시
- `POST /v1/decisions`: OpenAI Decisions API 공식 endpoint

API 키는 절대 브라우저 코드나 Git에 넣지 마세요.
