"""Run with python3 -m arena.test_premium_ui against the local arena on 8123."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT = Path(__file__).resolve().parents[3] / 'outputs'

def main():
    OUTPUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width, height in [(844,390), (667,375), (1280,800)]:
            page = browser.new_page(viewport={'width':width,'height':height})
            errors=[]
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto('http://127.0.0.1:8123/?pin=7777')
            page.wait_for_selector('#board-cards .board-slot')
            assert page.locator('#board-cards .board-slot').count()==5
            page.locator('#btn-new-hand').click()
            page.wait_for_selector('#seat-0 .card')
            page.wait_for_timeout(250)
            if height < 600:
                assert page.evaluate('document.documentElement.scrollHeight <= innerHeight'), 'vertical overflow'
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'horizontal overflow'
            # Board must not overlap any player, including Hero.
            overlaps=page.evaluate('''() => {
                const b=document.querySelector('.community-board-frame').getBoundingClientRect();
                return [...document.querySelectorAll('.player-seat')].filter(s=>{
                    const a=s.getBoundingClientRect();
                    return a.left<b.right && a.right>b.left && a.top<b.bottom && a.bottom>b.top;
                }).map(s=>s.id);
            }''')
            page.screenshot(path=str(OUTPUT/f'arena-{width}x{height}.png'))
            assert not overlaps, (width,height,overlaps)
            for button in page.locator('.btn-action').all():
                assert button.bounding_box()['height'] >= 44
            page.screenshot(path=str(OUTPUT/f'arena-{width}x{height}.png'))
            assert not errors, errors
            page.close()
        page=browser.new_page(viewport={'width':844,'height':390})
        states=[]
        page.on('websocket',lambda ws: ws.on('framereceived',lambda payload: states.append(json.loads(payload))))
        page.goto('http://127.0.0.1:8123/?pin=7777')
        page.wait_for_selector('#board-cards .board-slot')
        page.locator('#btn-speed-toggle').click()
        page.locator('#btn-new-hand').click()
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if states and states[-1]['hand_number'] and states[-1]['is_terminal']:
                break
            if page.locator('#btn-call').is_enabled():
                page.locator('#btn-call').click()
            page.wait_for_timeout(200)
        assert states[-1]['is_terminal'], 'hand did not complete'
        counts=[len(s['board_cards']) for s in states if s['hand_number']]
        assert all(c in (0,3,4,5) for c in counts), counts
        assert counts==sorted(counts), counts
        boards=[s['board_cards'] for s in states if s['hand_number']]
        assert all(b[:len(a)]==a for a,b in zip(boards,boards[1:])), boards
        for state in states:
            assert len(state['board_cards'])==len(set(state['board_cards']))
            if not state['is_terminal']:
                assert all(all(c=='??' for c in player['cards']) for player in state['players'] if player['seat']!=0)
        page.wait_for_timeout(450)
        page.screenshot(path=str(OUTPUT/'arena-showdown.png'))
        print('PASS: layouts, 44px targets, no overlap, live hand, card privacy; board counts:',sorted(set(counts)))
        browser.close()

if __name__=='__main__':
    main()
