import asyncio
from browser_use import Agent, Browser
from browser_use.llm.ollama.chat import ChatOllama

async def main() -> None:
    llm = ChatOllama(
        model="qwen2.5:7b",
    )
    
    agent = Agent(
        task="""
충남대학교 도서관 홈페이지(https://library.cnu.ac.kr)에 접속해서
통합검색창에 '자바의 정석'을 입력하고 검색 버튼을 클릭해.
검색 결과에 나오는 책 제목과 저자를 알려줘.
""",
        llm=llm,
        browser=Browser(),
        use_vision=False, # 스크린샷 안 보내서 훨씬 빠름(추가)
    )
    await agent.run(max_steps=10)

if __name__ == '__main__':
    asyncio.run(main())