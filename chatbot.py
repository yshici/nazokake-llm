from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from pydantic_settings import BaseSettings, SettingsConfigDict
import chainlit as cl
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    TAVILY_API_KEY: str

    LLM_MODEL_NAME: str = "gpt-4o-2024-05-13"
    FAST_LLM_MODEL_NAME: str = "gpt-3.5-turbo-0125"
    TAVILY_MAX_RESULTS: int = 5
    LANGCHAIN_TRACING_V2: str = "false"

settings = Settings()


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True, model=settings.LLM_MODEL_NAME)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは謎かけの名人です。"
                          "「〇〇とかけまして〜」という形式でお題が与えられます。"
                          "そのお題に対して、「XXと解きます。その心は、どちらも〜〜でしょう。」という形式で回答してください。",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
