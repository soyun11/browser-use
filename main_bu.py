from browser_use import Agent, Browser, ChatBrowserUse
# from browser_use import ChatGoogle  # ChatGoogle(model='gemini-3-flash-preview')
# from browser_use import ChatAnthropic  # ChatAnthropic(model='claude-sonnet-4-6')
import asyncio

async def main():
    browser = Browser(
        # use_cloud=True,  # Use a stealth browser on Browser Use Cloud
    )

    agent = Agent(
        task="충남대학교 도서관 홈페이지(https://library.cnu.ac.kr)에 접속해서 통합검색창에 '자바의 정석'을 입력하고 검색 버튼을 클릭해.검색 결과에 나오는 책 제목과 저자를 알려줘.",
        llm=ChatBrowserUse(),
        # llm=ChatGoogle(model='gemini-3-flash-preview'),
        # llm=ChatAnthropic(model='claude-sonnet-4-6'),
        browser=browser,
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())