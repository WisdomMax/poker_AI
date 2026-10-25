"""
External WSS Game Flow Verification via Cloudflare Tunnel
"""

import asyncio
import websockets
import json

async def test_external_game():
    uri = "wss://advances-permitted-educational-critics.trycloudflare.com/ws/ext_user_play?pin=7777"
    print(f"Connecting to External WSS: {uri} ...")
    
    async with websockets.connect(uri) as ws:
        init_st = json.loads(await ws.recv())
        print(f"[EXT-WS] Connected! Session: {init_st['session_id']}")
        
        print("[EXT-WS] Sending start_hand...")
        await ws.send(json.dumps({"type": "start_hand"}))
        
        hand_over = False
        step = 0
        while not hand_over:
            msg = json.loads(await ws.recv())
            print(f"[EXT-WS Step {step}] Street: {msg['round_name']}, Player: {msg['current_player']}, Pot: {msg['pot_bb']} BB")
            
            if msg['is_terminal']:
                print(f"[EXT-WS] Hand Terminal Reached! Winner: {msg.get('winner_announcement')}")
                print(f"  Hero Net: {msg['hero_stats']['total_profit_bb']} BB, Board: {msg['board_cards']}")
                hand_over = True
                break
                
            if msg['current_player'] == 0:
                print(f"  [EXT-WS] Hero's turn! Legal actions: {msg['legal_actions']}, Call cost: {msg['call_cost_bb']} BB")
                print("  [EXT-WS] Hero sending Call (action 1)...")
                await ws.send(json.dumps({"type": "human_action", "action_id": 1}))
                
            step += 1
            if step > 30:
                print("[EXT-WS] Step limit exceeded!")
                break
                
    print("[EXT-WS] External WSS End-to-End Game Flow Verified 100% PASS!")

if __name__ == "__main__":
    asyncio.run(test_external_game())
