import os

def clean():
    with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v54_instruction.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = '# POKER AI V5.4'
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Start marker not found")
        return

    prompt = content[start_idx:]
    # GenUI 끝부분 정리
    # 문자열 내의 GenUI 닫는 괄호 탐색
    for tail_marker in ['`)}', '")}', '`\n)}']:
        t_idx = prompt.rfind(tail_marker)
        if t_idx != -1:
            prompt = prompt[:t_idx]
            break

    os.makedirs('/Volumes/NVME/7.AI_vibe_coding/hangame/v54', exist_ok=True)
    os.makedirs('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/data', exist_ok=True)
    os.makedirs('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/models', exist_ok=True)
    os.makedirs('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/logs', exist_ok=True)

    with open('/Volumes/NVME/7.AI_vibe_coding/hangame/v54/v54_instruction_clean.txt', 'w', encoding='utf-8') as f_out:
        f_out.write(prompt)
    print(f"Clean prompt saved: {len(prompt)} chars")

if __name__ == '__main__':
    clean()
