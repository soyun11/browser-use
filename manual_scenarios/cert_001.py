import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://cnu.icerti.com/icerti/index_internet.jsp?t=3415")
    page.get_by_role("link", name="증명서 발급 바로가기").click()
    page.get_by_role("textbox", name="아이디(학번)").click()
    page.get_by_role("textbox", name="아이디(학번)").fill("202302554")
    page.get_by_role("textbox", name="아이디(학번)").press("Tab")
    page.get_by_title("비밀번호", exact=True).press("Tab")
    page.get_by_role("textbox", name="비밀번호").fill("jessica0998!@")
    page.get_by_role("textbox", name="비밀번호").press("Enter")
    page.get_by_role("button", name="확인").click()
    page.get_by_text("이용약관, 개인정보 수집 및 이용에 동의합니다").click()
    page.get_by_role("button", name="동의", exact=True).click()
    page.get_by_role("button", name="신청").click()
    page.get_by_role("button", name=" 프린트/이메일").click()
    page.locator("#CertCounter9 > .input-group-btn > .ko.counter-plus").click()
    page.get_by_role("button", name="확인").click()
    page.get_by_role("button", name="확인").click()
    page.get_by_role("button", name="이메일").first.click()
    page.locator("iframe[name=\"cert_frame\"]").content_frame.get_by_text("확인 취소").click()
    page.locator("iframe[name=\"cert_frame\"]").content_frame.get_by_role("button", name="확인").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
