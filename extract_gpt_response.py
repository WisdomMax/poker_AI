#!/usr/bin/env python3
"""
POKER AI - SILENT BACKGROUND CHATGPT EXTRACTOR (extract_gpt_response.py)
- ChatGPT 창을 화면에 전혀 띄우지 않고(No Activate, No Focus, No Cliclick)
- macOS Accessibility C API (AXUIElement)를 통해 메모리 트리에서 직접 최신 답변/지침을 100% 백그라운드 추출
"""

import os
import sys
import ctypes
from ctypes import c_void_p, c_char_p, byref, POINTER
from AppKit import NSWorkspace

OUTPUT_INSTRUCTION_PATH = "/Volumes/NVME/7.AI_vibe_coding/hangame/latest_gpt_instruction.txt"

def get_chatgpt_pid():
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if 'chatgpt' in app.localizedName().lower() and app.activationPolicy() == 0:
            return app.processIdentifier()
    return None

def extract_latest_gpt_response(marker_keyword: str = "POKER AI V5.3.2"):
    pid = get_chatgpt_pid()
    if not pid:
        print("[오류] 실행 중인 ChatGPT 메인 프로세스를 찾지 못했습니다.")
        return None

    app_services = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
    cf = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')

    AXUIElementCreateApplication = app_services.AXUIElementCreateApplication
    AXUIElementCreateApplication.argtypes = [ctypes.c_int]
    AXUIElementCreateApplication.restype = c_void_p

    AXUIElementCopyAttributeValue = app_services.AXUIElementCopyAttributeValue
    AXUIElementCopyAttributeValue.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
    AXUIElementCopyAttributeValue.restype = ctypes.c_int

    CFStringCreateWithCString = cf.CFStringCreateWithCString
    CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, ctypes.c_uint32]
    CFStringCreateWithCString.restype = c_void_p

    CFArrayGetCount = cf.CFArrayGetCount
    CFArrayGetCount.argtypes = [c_void_p]
    CFArrayGetCount.restype = ctypes.c_long

    CFArrayGetValueAtIndex = cf.CFArrayGetValueAtIndex
    CFArrayGetValueAtIndex.argtypes = [c_void_p, ctypes.c_long]
    CFArrayGetValueAtIndex.restype = c_void_p

    CFStringGetCString = cf.CFStringGetCString
    CFStringGetCString.argtypes = [c_void_p, c_char_p, ctypes.c_long, ctypes.c_uint32]
    CFStringGetCString.restype = ctypes.c_bool

    CFGetTypeID = cf.CFGetTypeID
    CFGetTypeID.argtypes = [c_void_p]
    CFGetTypeID.restype = ctypes.c_ulong

    CFStringGetTypeID = cf.CFStringGetTypeID
    CFStringGetTypeID.restype = ctypes.c_ulong

    def cf_to_str(cf_val):
        if not cf_val: return ''
        if CFGetTypeID(cf_val) != CFStringGetTypeID(): return ''
        buf = ctypes.create_string_buffer(65536)
        if CFStringGetCString(cf_val, buf, 65536, 0x08000100):
            return buf.value.decode('utf-8', errors='ignore')
        return ''

    def make_cf(s):
        return CFStringCreateWithCString(None, s.encode('utf-8'), 0x08000100)

    kAXWindows = make_cf('AXWindows')
    kAXChildren = make_cf('AXChildren')
    kAXValue = make_cf('AXValue')
    kAXRole = make_cf('AXRole')

    app_ref = AXUIElementCreateApplication(pid)
    all_texts = []

    def traverse(elem, depth=0):
        if depth > 35: return
        r_val = c_void_p()
        if AXUIElementCopyAttributeValue(elem, kAXRole, byref(r_val)) == 0 and r_val.value:
            role = cf_to_str(r_val)
            if role == 'AXStaticText':
                v_val = c_void_p()
                if AXUIElementCopyAttributeValue(elem, kAXValue, byref(v_val)) == 0 and v_val.value:
                    s = cf_to_str(v_val)
                    if s: all_texts.append(s)
        children = c_void_p()
        if AXUIElementCopyAttributeValue(elem, kAXChildren, byref(children)) == 0 and children.value:
            c_count = CFArrayGetCount(children)
            for i in range(c_count):
                traverse(CFArrayGetValueAtIndex(children, i), depth+1)

    win_arr = c_void_p()
    if AXUIElementCopyAttributeValue(app_ref, kAXWindows, byref(win_arr)) == 0 and win_arr.value:
        traverse(CFArrayGetValueAtIndex(win_arr, 0))

    full_doc = '\n'.join(all_texts)
    print(f"[1] 백그라운드 메모리 탐색 완료 (총 {len(full_doc):,} 자 추출)")

    pos = full_doc.rfind(marker_keyword)
    if pos != -1:
        target_text = full_doc[pos:].strip()
        print(f"[2] 마커 '{marker_keyword}' 기준 분리 완료 ({len(target_text):,} 자)")
    else:
        target_text = full_doc[-10000:].strip()
        print(f"[2] 최근 10,000자 기준 분리 완료")

    with open(OUTPUT_INSTRUCTION_PATH, "w", encoding="utf-8") as f:
        f.write(target_text)
    print(f"[3] 최신 지침 백그라운드 저장 완료: {OUTPUT_INSTRUCTION_PATH}\n")
    return target_text

if __name__ == '__main__':
    kw = sys.argv[1] if len(sys.argv) > 1 else "POKER AI V5.3.2"
    extract_latest_gpt_response(kw)
