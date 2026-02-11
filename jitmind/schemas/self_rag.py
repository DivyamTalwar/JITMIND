# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel, Field


class SelfRAGDecision(BaseModel):
    ISREL: bool = Field(..., description="Retrieved content relevant?")
    ISSUP: bool = Field(..., description="Response supported by evidence?")
    ISUSE: bool = Field(..., description="Response useful?")

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        schema = super().model_json_schema()
        schema["required"] = ["ISREL", "ISSUP", "ISUSE"]
        schema["additionalProperties"] = False
        return schema
