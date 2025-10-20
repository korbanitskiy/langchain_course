from pydantic import BaseModel, Field


class Source(BaseModel):
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    sources: list[Source] = Field(description="The sources used to answer the question", default_factory=list)
