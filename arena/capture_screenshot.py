"""
Capture screenshot of POKER AI HUMAN ARENA web interface
"""

import asyncio
from playwright.async_api import async_playwright

async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # 데스크톱 해상도 (1440x900)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto("http://localhost:8000", wait_until="networkidle")
        await asyncio.sleep(2.0)
        
        # 새 핸드 시작 버튼 클릭
        new_hand_btn = page.locator("#btn-new-hand")
        if await new_hand_btn.is_enabled():
            print("Clicking Deal Next Hand...")
            await new_hand_btn.click()
            await asyncio.sleep(2.0)
            
        screenshot_path1 = "arena/static/screenshot.png"
        screenshot_path2 = "/Users/wisdom/.gemini/antigravity/brain/0d2be791-2346-4033-9414-1e2d873e986b/arena_gameplay_screenshot.png"
        
        await page.screenshot(path=screenshot_path1)
        await page.screenshot(path=screenshot_path2)
        print(f"Screenshots successfully saved to:\n  - {screenshot_path1}\n  - {screenshot_path2}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture())
