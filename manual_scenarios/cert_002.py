import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://cnu.icerti.com/icerti/index_internet.jsp?t=3415")
    page.get_by_role("link", name="증명서 발급 바로가기").click()
    page.get_by_role("textbox", name="아이디(학번)").fill("202302554")
    page.get_by_role("textbox", name="비밀번호").click()
    page.get_by_role("textbox", name="비밀번호").fill("jessica0998!@")
    page.get_by_role("button", name="로그인").click()
    page.get_by_text("학적 등록 정보 안내 보기").click()
    page.get_by_role("button", name="확인").click()
    page.get_by_role("checkbox", name="이용약관, 개인정보 수집 및 이용에 동의합니다").check()
    page.get_by_role("button", name="동의", exact=True).click()
    page.get_by_role("heading", name="증명서 신청 확인").click()
    page.get_by_role("button", name="신청").click()
    page.once("dialog", lambda dialog: dialog.dismiss())
    page.get_by_role("button", name=" PDF").click()
    page.locator("input[name=\"G0017,ko_KR,null\"]").check()
    page.get_by_role("button", name="확인").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
