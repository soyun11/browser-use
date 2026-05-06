import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://library.cnu.ac.kr/")
    page.get_by_role("textbox", name="검색어 입력").click()
    page.get_by_role("textbox", name="검색어 입력").fill("파이썬")
    page.get_by_role("button", name="Submit").click()
    page.locator("#catalogs").get_by_role("link", name="더보기", exact=True).click()
    page.get_by_role("link", name="항목선택").click()
    page.locator("a").filter(has_text="출판년").click()
    page.get_by_role("link", name="정렬").click()
    page.locator("a").filter(has_text="내림차순").click()
    page.get_by_role("link", name="10").dblclick()
    page.get_by_text("5", exact=True).nth(2).click()
    page.get_by_role("button", name="조회").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
