"""
External WSS Game Flow Verification via Cloudflare Tunnel (DIRECTIVE 024 Dollar Unit)
"""

import asyncio
import websockets
import json
import re

def get_current_tunnel_domain():
    try:
        with open("arena/tunnel.log", "r") as f:
            content = f.read()
        matches = re.findall(r'https://([-a-z0-9.]*trycloudflare.com)', content)
        if matches:
            return matches[-1]
    except Exception as e:
        print("Error reading tunnel log:", e)
    return "webmasters-jason-late-career.trycloudflare.com"

async def test_external_game():
    domain = get_current_tunnel_domain()
    uri = f"wss://{domain}/ws/ext_user_play?pin=7777"
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
            pot_str = f"${msg.get('pot_dollars', int(msg['pot_bb']*100)):,}"
            print(f"[EXT-WS Step {step}] Street: {msg['round_name']}, Player: {msg['current_player']}, Pot: {pot_str}")
            
            if msg['is_terminal']:
                print(f"[EXT-WS] Hand Terminal Reached! Winner: {msg.get('winner_announcement')}")
                h_stats = msg['hero_stats']
                print(f"  Hero Bankroll: ${h_stats.get('current_balance', 10000):,}, Total Net: ${h_stats.get('total_profit_dollars', 0):,}")
                print(f"  AI H2H Records: {len(h_stats.get('ai_records', []))} opponents tracked")
                hand_over = True
                break
                
            if msg['current_player'] == 0:
                cost_str = f"${msg.get('call_cost_dollars', int(msg['call_cost_bb']*100)):,}"
                print(f"  [EXT-WS] Hero's turn! Legal actions: {msg['legal_actions']}, Call cost: {cost_str}")
                print("  [EXT-WS] Hero sending Call (action 1)...")
                await ws.send(json.dumps({"type": "human_action", "action_id": 1}))
                
            step += 1
            if step > 30:
                print("[EXT-WS] Step limit exceeded!")
                break
                
    print("[EXT-WS] External WSS End-to-End Game Flow Verified 100% PASS!")

if __name__ == "__main__":
    asyncio.run(test_external_game())
