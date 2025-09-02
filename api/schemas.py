from pydantic import BaseModel, Field

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=30)
    max_length: int = 128
    min_length: int = 32

class SummarizeResponse(BaseModel):
    summary: str
