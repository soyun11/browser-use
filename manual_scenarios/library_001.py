import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://library.cnu.ac.kr/")
    page.get_by_role("textbox", name="검색어 입력").click()
    page.get_by_role("textbox", name="검색어 입력").fill("자바의 정석")
    page.get_by_role("button", name="Submit").click()
    page.locator("#catalogs").get_by_role("link", name="더보기", exact=True).click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
