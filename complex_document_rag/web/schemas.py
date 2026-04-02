"""网页路由共用的 Pydantic 数据模型。"""

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """前端会话历史中保存的一轮用户/助手对话。"""
    query: str = Field(..., min_length=1, description="历史用户问题")
    answer: str = Field(default="", description="历史助手回答")


class QueryRequest(BaseModel):
    """同步查询与流式查询接口共用的请求体。"""
    query: str = Field(..., min_length=1, description="用户问题")
    generate_answer: bool = Field(default=True, description="是否生成最终回答")
    history: list[ConversationTurn] = Field(default_factory=list, description="最近几轮对话历史")
