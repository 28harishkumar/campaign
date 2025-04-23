import json
import re
import sqlite3
import random
import os
import uuid
from datetime import datetime
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import Graph, StateGraph
from .base import CampaignBase, CampaignState


class CampaignAnalyzer(CampaignBase):
    def __init__(
        self,
        config: Dict,
        details: Dict,
        model_name: str = "anthropic",
        model_version: str = "claude-3-5-haiku-20241022",
    ):
        super().__init__(model_name, model_version)
        self.config = config
        self.details = details
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")

        with open(os.path.join(prompt_dir, "analyze_results.txt"), "r") as f:
            self.analyze_results_prompt = f.read()

        with open(os.path.join(prompt_dir, "validate_results.txt"), "r") as f:
            self.validate_results_prompt = f.read()

        with open(os.path.join(prompt_dir, "roi_calculation.txt"), "r") as f:
            self.roi_calculation_prompt = f.read()

    def stage3_generate_results(self, state: CampaignState) -> CampaignState:
        """Stage 3: Generate experiment results"""
        print("Stage 3: Generate experiment results")

        state["details"] = self.details
        state["config"] = self.config

        if not state["details"]:
            raise ValueError(
                "Please upload campaign first to generate experiment details"
            )

        # Connect to SQLite database
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()

        # Generate 10 random events for each experiment
        for experiment in state["details"]["experiments"]:
            experiment_id = experiment["experiment_id"]

            # Generate 10 random users for this experiment
            for i in range(10):
                user_id = str(uuid.uuid4())

                # Insert user if not exists
                cursor.execute(
                    "INSERT OR IGNORE INTO users (user_id, signup_date, last_active, user_segment) VALUES (?, ?, ?, ?)",
                    (
                        user_id,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        "test",
                    ),
                )

                # Generate random wallet visits (1-3 visits per user)
                num_visits = random.randint(1, 3)
                for _ in range(num_visits):
                    cursor.execute(
                        "INSERT INTO wallet_visits (user_id, timestamp, experiment_id) VALUES (?, ?, ?)",
                        (user_id, datetime.now().isoformat(), experiment_id),
                    )

                # Generate random deposit events (0-2 deposits per user)
                num_deposits = random.randint(0, 2)
                for _ in range(num_deposits):
                    # Generate random amount between min_deposit and 2x min_deposit
                    amount = random.uniform(0, 100000)

                    cursor.execute(
                        "INSERT INTO deposit_events (user_id, amount, timestamp, experiment_id, payment_method) VALUES (?, ?, ?, ?, ?)",
                        (
                            user_id,
                            amount,
                            datetime.now().isoformat(),
                            experiment_id,
                            "credit_card",
                        ),
                    )

        # Commit changes and close connection
        conn.commit()
        conn.close()

        results = []
        for i in range(state["details"]["number_of_experiments"]):
            experiment = state["details"]["experiments"][i]
            experiment_id = experiment["experiment_id"]

            # Calculate ROI using the SQL query from config
            try:
                conn = sqlite3.connect(self.database_file)
                cursor = conn.cursor()
                query_results = {}

                for query in state["config"]["sql_query"]:
                    cursor.execute(query.format(experiment_id))
                    query_results[query] = cursor.fetchall()
                conn.close()

                roi_calculation_system_prompt = self.roi_calculation_prompt.format(
                    query_results=query_results,
                    roi_calculation=state["config"]["roi_calculation"],
                )

                # Execute the ROI calculation query
                response = self.llm.invoke(
                    [
                        SystemMessage(content=roi_calculation_system_prompt),
                        HumanMessage(
                            content=f"Query results: {query_results}\nExperiment ID: {experiment_id}"
                        ),
                    ]
                )

                code_block = re.search(
                    r"```json\n(.*?)\n```", response.content, re.DOTALL
                )
                if not code_block:
                    raise ValueError("No JSON code block found in response")

                roi_calculation_parser = JsonOutputParser()
                roi_calculation = roi_calculation_parser.parse(code_block.group(1))

                print("ROI calculation: ", roi_calculation)
                roi = roi_calculation["roi"]
                thought_process = roi_calculation["thought_process"]

            except sqlite3.Error as e:
                print(f"Error calculating ROI for experiment {experiment_id}: {str(e)}")
                roi = 0.0  # Default ROI on error

            experiment["roi"] = roi
            experiment["timestamp"] = datetime.now().isoformat()
            results.append(experiment)

        state["results"] = results

        # Save results to experiment_history.json
        self._save_history(results)

        print("Generated results:", json.dumps(results, indent=2))
        return state

    def stage4_analyze_results(self, state: CampaignState) -> CampaignState:
        """Stage 4: Analyze experiment results"""
        print("Stage 4: Analyze experiment results")

        if not state["results"]:
            raise ValueError("Please run stage 3 first to generate results")

        system_prompt = self.analyze_results_prompt.format(
            config=json.dumps(state["config"]),
            details=json.dumps(state["details"]),
            results=json.dumps(state["results"]),
        )

        state["analyze_results_prompt"] = system_prompt

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content="Analyze the results and suggest new experiments with statistical justification"
            ),
        ]

        response = self.llm.invoke(messages)
        code_block = re.search(r"```json\n(.*?)\n```", response.content, re.DOTALL)
        if not code_block:
            raise ValueError("No JSON code block found in response")

        parser = JsonOutputParser()
        analysis = parser.parse(code_block.group(1))

        state["analysis"] = analysis["analysis"]

        # Create history entry
        history_entry = {
            "config": state["config"],
            "details": state["details"],
            "results": state["results"],
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to history
        # self._save_history(history_entry)

        # Prepare for next cycle with improved context
        state["current_stage"] = 3
        state["prompt"] = (
            f"""Based on the following analysis: {json.dumps(analysis['analysis'], indent=2)}
            
            Previous Configuration: {json.dumps(state["config"], indent=2)}

            Previous Experiment Details: {json.dumps(state["details"], indent=2)}

            Previous Results: {json.dumps(state["results"], indent=2)}

            Analysis: {json.dumps(state["analysis"], indent=2)}
            
            Generate a new experiment configuration that:
            1. Addresses the learnings from previous experiments
            2. Improves upon the ROI metric: {state["config"]["roi"]}
            3. Considers statistical significance
            4. Builds upon successful variables from previous experiments
            5. Run new experiments suggested by the analysis
            """
        )

        print("Cycle prompt for the next cycle: \n", state["prompt"])
        return state

    def validate_results(self, state: CampaignState) -> CampaignState:
        """Validate the results"""
        print("Stage 4: Validate the results")
        validation_errors = []
        itteration = 0

        while itteration < 3:
            itteration += 1

            # TODO: while keeping best 2 or 1(if total experiments are less then 3)

            for ex_id in state["analysis"]["successful_experiments"]:
                ex = list(filter(lambda x: x["experiment_id"] == ex_id, self.history))
                if ex and ex[0]["roi"] < state["config"]["roi"]:
                    validation_errors.append(
                        f"Experiment {ex_id} has ROI {ex[0]['roi']} which is less than targeted ROI {state['config']['roi']}"
                    )

            if len(validation_errors) > 0:
                system_prompt = self.validate_results_prompt.format(
                    config=json.dumps(state["config"]),
                    last_prompt=state["analyze_results_prompt"],
                    details=json.dumps(state["details"]),
                    results=json.dumps(state["results"]),
                    last_analysis=json.dumps(state["analysis"], indent=2),
                    validation_errors="\n".join(validation_errors),
                )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content="Analyze the results and suggest new experiments with statistical justification"
                    ),
                ]

                response = self.llm.invoke(messages)
                code_block = re.search(
                    r"```json\n(.*?)\n```", response.content, re.DOTALL
                )
                if not code_block:
                    raise ValueError("No JSON code block found in response")

                parser = JsonOutputParser()
                analysis = parser.parse(code_block.group(1))

                print("Last cycle Analysis: \n", analysis)
            else:
                break
        return state

    def create_graph(self) -> Graph:
        """Create the langgraph workflow"""
        workflow = StateGraph(CampaignState)

        # Add nodes
        workflow.add_node("stage3", self.stage3_generate_results)
        workflow.add_node("stage4", self.stage4_analyze_results)
        workflow.add_node("stage5", self.validate_results)

        # Add edges
        workflow.add_edge("stage3", "stage4")
        workflow.add_edge("stage4", "stage5")

        # Set entry point
        workflow.set_entry_point("stage3")

        return workflow.compile()
