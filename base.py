import json
import os
import re
from typing import Dict, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import random
import argparse
from datetime import datetime
from langgraph.graph import Graph, StateGraph
import sqlite3


# Type definitions
class ExperimentConfig(TypedDict):
    no_of_users: int
    experiment_variables: List[str]
    roi: str
    experiment_duration: str


class ExperimentDetails(TypedDict):
    number_of_experiments: int
    experiments: List[Dict[str, str]]


class AnalysisResult(TypedDict):
    analysis: str
    new_possible_experiment: List[Dict[str, str]]


class ExperimentResult(TypedDict):
    experiment_id: str
    variables: Dict[str, str]
    roi: float
    timestamp: str


class ExperimentHistory(TypedDict):
    config: ExperimentConfig
    details: ExperimentDetails
    results: List[ExperimentResult]
    analysis: AnalysisResult
    timestamp: str


class Analysis(TypedDict):
    new_possible_experiment: List[Dict[str, str]]
    successful_experiments: List[str]
    statistical_significance: str
    roi_comparison: str
    key_findings: str
    limitations: str
    recommendations: str
    terminate_experiments: List[str]
    thought_process: Dict[str, str]


class CampaignState(TypedDict):
    current_stage: int
    config: Optional[ExperimentConfig]
    details: Optional[ExperimentDetails]
    results: List[ExperimentResult]
    history: List[ExperimentHistory]
    analysis: Optional[AnalysisResult]
    prompt: Optional[str]
    generate_experiments_prompt: Optional[str]
    analyze_results_prompt: Optional[str]


class CampaignBase:
    def __init__(
        self,
        model_name: str = "anthropic",
        model_version: str = "claude-3-5-haiku-20241022",
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.llm = self._initialize_llm()
        self.history_file = "campaign/experiment_history.json"
        self.database_schema_file = "campaign/database_schema.json"
        self.database_file = "campaign/database.db"
        self.history = self._load_history().get("experiments", [])

    def extract_code_block(self, text):
        """
        Extracts json block from the text
        """
        pattern = r"```(?:json\n)?([\s\S]*?)```"
        blocks = [b.strip() for b in re.findall(pattern, text)]

        try:
            if blocks:
                return blocks[0]
            else:
                return None
        except Exception as e:
            print(f"Error extracting code block: {str(e)}")
            print(f"Text: {text}")
            return text

    def _initialize_llm(self):
        if self.model_name == "ollama":
            return ChatOllama(
                model=self.model_version,
                base_url="http://localhost:11434",
                temperature=0.3,
                top_p=0.9,
                num_ctx=4096,
            )
        elif self.model_name == "anthropic":
            # return ChatAnthropic(
            #     model=self.model_version,
            #     anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            #     temperature=0.3,
            #     top_p=0.9,
            # )
            return ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                temperature=0.7,
                max_tokens=1024,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        elif self.model_name == "openai":
            return ChatOpenAI(model=self.model_version)
        elif self.model_name == "gemini":
            return ChatGoogleGenerativeAI(model=self.model_version)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _load_history(self) -> List[Dict]:
        """Load existing history from file or return empty list if file doesn't exist"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(
                    "Warning: experiment_history.json is corrupted. Starting with empty history."
                )
                return {"experiments": []}
        return {"experiments": []}

    def _load_history_database(self) -> List[Dict]:
        """Load existing history from database or return empty list if database doesn't exist"""
        if os.path.exists(self.database_file):
            try:
                with sqlite3.connect(self.database_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM experiments")
                    return cursor.fetchall()
            except sqlite3.Error as e:
                print(f"Error loading history from database: {str(e)}")
                return []
        return []

    def _save_history(self, history_entry: Dict):
        """Append new entry to history and save to file"""
        self.history.extend(history_entry)
        try:
            with open(self.history_file, "w") as f:
                json.dump({"experiments": self.history}, f, indent=2)
            print(f"\nResults saved to {self.history_file}")
        except Exception as e:
            print(f"Error saving to {self.history_file}: {str(e)}")

    def _load_database_schema(self) -> str:
        """Load database schema from file or return empty string if file doesn't exist"""
        if os.path.exists(self.database_schema_file):
            try:
                with open(self.database_schema_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading database schema: {str(e)}")
                return ""
        return ""
