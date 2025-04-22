import os
import re
import json
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from .base import CampaignBase, CampaignState


class CampaignGenerator(CampaignBase):
    def __init__(
        self,
        model_name: str = "anthropic",
        model_version: str = "claude-3-5-haiku-20241022",
    ):
        super().__init__(model_name, model_version)
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        with open(os.path.join(prompt_dir, "generate_config.txt"), "r") as f:
            self.generate_config_prompt = f.read()
        with open(os.path.join(prompt_dir, "validate_sql.txt"), "r") as f:
            self.validate_sql_prompt = f.read()
        with open(os.path.join(prompt_dir, "generate_experiments.txt"), "r") as f:
            self.generate_experiments_prompt = f.read()
        with open(os.path.join(prompt_dir, "validate_experiments.txt"), "r") as f:
            self.validate_experiments_prompt = f.read()

    def stage1_generate_config(self, state: CampaignState) -> CampaignState:
        """Stage 1: Generate experiment configuration from user prompt"""
        print("Stage 1: Generate experiment configuration from user prompt")
        # Add historical context to the prompt only if history exists
        self.history = self._load_history().get("experiments", [])
        self.history_data = self._load_history_database()
        database_schema = self._load_database_schema()

        system_prompt = self.generate_config_prompt.format(
            database_schema=database_schema
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["prompt"]),
        ]

        response = self.llm.invoke(messages)
        try:
            # Extract the first code block from the response
            code_block = re.search(r"```json\n(.*?)\n```", response.content, re.DOTALL)
            if not code_block:
                raise ValueError("No JSON code block found in response")

            parser = JsonOutputParser()
            config = parser.parse(code_block.group(1))
            if not config or config == {}:
                raise ValueError("Empty or invalid JSON response received")
            state["config"] = config
            print("Generated config:", json.dumps(config, indent=2))
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            print("Raw response:", response.content)
        return state

    def stage1_validate_config(self, state: CampaignState) -> CampaignState:
        """Stage 1: Validate experiment configuration"""
        print("Stage 1: Validate experiment configuration")

        iteration = 0

        while iteration < 3:
            iteration += 1
            validation_errors = []

            try:
                if not state["config"]:
                    validation_errors.append(
                        "Please run stage 1 first to generate configuration"
                    )

                # Validate the config
                if state["config"]["no_of_users"] < 100:
                    validation_errors.append("No of users must be greater than 100")

                if not state["config"]["experiment_variables"]:
                    validation_errors.append("Experiment variables must be specified")

                if state["config"]["experiment_duration"] < 1:
                    validation_errors.append(
                        "Experiment duration must be greater than 1 day"
                    )

                if not state["config"]["roi"]:
                    validation_errors.append("ROI must be specified")

                if not state["config"]["sql_query"]:
                    validation_errors.append("SQL query must be specified")

                # check if sql query is valid and read only
                try:
                    prompt = self.validate_sql_prompt.format(
                        sql_query=state["config"]["sql_query"]
                    )
                    response = self.llm.invoke(prompt)
                    parser = JsonOutputParser()
                    code_block = re.search(
                        r"```json\n(.*?)\n```", response.content, re.DOTALL
                    )
                    if not code_block:
                        raise ValueError("No JSON code block found in response")

                    result = parser.parse(code_block.group(1))

                    if not result or result == {}:
                        validation_errors.append("Invalid sql query response")

                    if not result["valid"]:
                        validation_errors.append(result["error"])
                except Exception as e:
                    validation_errors.append("SQL query is not valid or read only")
            except Exception as e:
                print(f"Error parsing JSON response: {str(e)}")
                validation_errors.append(f"Error parsing LLM response: {str(e)}")

            for index, variable in enumerate(state["config"]["experiment_variables"]):
                v_range = input(
                    f"Enter the range for {variable['name']}: (default:{variable.get("variable_range")})"
                )

                if v_range:
                    state["config"]["experiment_variables"][index][
                        "variable_range"
                    ] = v_range

                if variable.get("type") and variable["type"].startswith("array"):
                    state["config"]["experiment_variables"][index][
                        "number_of_values"
                    ] = input(f"Enter the number of values for {variable['name']}: ")

                if variable.get("type") and (
                    variable["type"] == "decimal" or variable["type"] == "integer"
                ):
                    state["config"]["experiment_variables"][index]["format"] = input(
                        f"Enter the format for {variable['name']}: "
                    )

            if state["config"]["currency"] == "" or state["config"]["currency"] == None:
                state["config"]["currency"] = input("Enter the currency: ")
            else:
                val = input(
                    f"Confirm the currency: {state['config']['currency']} Yes/No or write your own currency: "
                )

                if val and val.lower() == "no":
                    state["config"]["currency"] = input("Enter the currency: ")
                elif val and val.lower() != "yes" and val.lower() != "y":
                    state["config"]["currency"] = val

            if not validation_errors:
                break

            # If there are validation errors, send them back to the LLM for correction
            if validation_errors:
                error_message = "\n".join(validation_errors)
                system_prompt = f"""You are an expert in experiment design and A/B testing for digital products. 
                Your previous response had the following validation errors:
                
                {error_message}
                
                Original prompt: {state["prompt"]}
                Previous LLM response: {json.dumps(state["config"], indent=2)}

                You MUST return only a valid JSON object in this exact format. Do not return anything else.
                """

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=state["prompt"]),
                ]

                response = self.llm.invoke(messages)
                try:
                    # Extract the first code block from the response
                    code_block = re.search(
                        r"```json\n(.*?)\n```", response.content, re.DOTALL
                    )
                    if not code_block:
                        raise ValueError("No JSON code block found in response")

                    parser = JsonOutputParser()
                    config = parser.parse(code_block.group(1))
                    if not config or config == {}:
                        validation_errors.append(
                            "Empty or invalid JSON response received"
                        )
                    state["config"] = config
                    print("Generated corrected config:", json.dumps(config, indent=2))
                except Exception as e:
                    print(f"Error parsing JSON response: {str(e)}")
                    print("Raw response:", response.content)

        return state

    def confirm_state1(self, state: CampaignState) -> CampaignState:
        """Stage 1 confirmation: Confirm generated experiment configuration from user prompt"""
        print(
            "Stage 1 confirmation: Confirm generated experiment configuration from user prompt"
        )
        # Add historical context to the prompt only if history exists

        print(json.dumps(state["config"], indent=2))

        user_response = input(
            "\nDo you want to continue with the generated configuration? (yes/describe the changes you want to make): "
        ).lower()

        database_schema = self._load_database_schema()

        system_prompt = f"""You are an expert in experiment design and A/B testing for app and websites. 
        Your task is to parse natural language input from the user and give me output in the specified output JSON format below.
        
        Previous LLM Prompt: {state["prompt"]}

        Previous LLM response: {json.dumps(state["config"], indent=2)}
        User want to make the following changes: {user_response}

        You MUST return only a valid JSON object in this exact format (with codeblock). Do not return anything else.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["prompt"]),
        ]

        if user_response in ["yes", "y"]:
            state["is_config_confirmed"] = True
            return state

        state["is_config_confirmed"] = False
        response = self.llm.invoke(messages)
        try:
            parser = JsonOutputParser()
            config = parser.parse(response.content)
            if not config or config == {}:
                raise ValueError("Empty or invalid JSON response received")
            state["config"] = config
            print("Generated config:", json.dumps(config, indent=2))
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            print("Raw response:", response.content)
        return state

    # def stage1_prompt(self, state: CampaignState) -> CampaignState:
    #     """Stage 1: Prompt for experiment configuration"""
    #     print("Stage 1: Prompt for experiment configuration")
    #     # validate and take input for variable range, format and number of values
    #     return state

    def stage2_generate_experiments(self, state: CampaignState) -> CampaignState:
        """Stage 2: Generate experiment details based on configuration"""

        print("Stage 2: Generate experiment details based on configuration")

        if not state["config"]:
            raise ValueError("Please run stage 1 first to generate configuration")

        self.history_data = self._load_history_database()

        # Add historical context to the prompt
        historical_context = ""

        if self.history:
            historical_context = f"""
            Previous Experiment History:
            {json.dumps(self.history, indent=2)}

            Previous Experiment Data:
            {json.dumps(self.history_data, indent=2)}
            
            Key Learnings from Previous Experiments:
            1. Successful Configurations: Identify which user counts and durations yielded good results
            2. Effective Variables: Note which experiment variables showed significant impact
            3. Optimal Sample Sizes: Consider what sample sizes provided statistically significant results
            4. Duration Insights: Review what experiment durations were most effective
            5. ROI Patterns: Analyze which configurations led to better ROI outcomes
            
            Use these insights to:
            - Build upon successful strategies
            - Avoid previous pitfalls
            - Optimize for better results
            """

        range_prompt = ""
        for variable in state["config"]["experiment_variables"]:
            range = variable.get("variable_range").split("-")
            range_prompt += f"value of {variable['name']} should be more than or equal to {range[0]} and less than or equal to {range[1]}\n"

        system_prompt = self.generate_experiments_prompt.format(
            config=json.dumps(state["config"]),
            roi=state["config"]["roi"],
            historical_context=historical_context,
            range_prompt=range_prompt,
        )

        state["generate_experiments_prompt"] = system_prompt

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content="Generate experiment details with proper statistical controls"
            ),
        ]

        response = self.llm.invoke(messages)
        try:
            # Extract the first code block from the response
            code_block = re.search(r"```json\n(.*?)\n```", response.content, re.DOTALL)
            if not code_block:
                raise ValueError("No JSON code block found in response")

            parser = JsonOutputParser()
            details = parser.parse(code_block.group(1))
            if not details or details == {}:
                raise ValueError("Empty or invalid JSON response received")
            state["details"] = details
            print("Generated details:", json.dumps(details, indent=2))
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            print("Raw response:", response.content)
            raise
        return state

    def stage2_validate_experiments(self, state: CampaignState) -> CampaignState:
        """Stage 2: Validate experiment details"""
        print("Stage 2: Validate experiment details")

        iteration = 0

        while iteration < 3:
            validation_errors = []

            try:
                if not state["details"]:
                    validation_errors.append(
                        "Please run stage 2 first to generate experiment details"
                    )

                # Validate the details
                if state["details"]["number_of_experiments"] < 1:
                    validation_errors.append(
                        "Number of experiments must be greater than 0"
                    )

                for experiment in state["details"]["experiments"]:
                    if not experiment["experiment_id"]:
                        validation_errors.append("Experiment ID must be specified")

                    if not experiment["variable_values"]:
                        validation_errors.append("Variable values must be specified")

                    if not experiment["hypothesis"]:
                        validation_errors.append("Hypothesis must be specified")

                    if not experiment["expected_impact"]:
                        validation_errors.append("Expected impact must be specified")

                    if not experiment["sample_size"]:
                        validation_errors.append("Sample size must be specified")

                    if not experiment["duration"]:
                        validation_errors.append("Duration must be specified")

                    if experiment["duration"] < 1:
                        validation_errors.append("Duration must be greater than 1 day")

                    if experiment["sample_size"] < 100:
                        validation_errors.append("Sample size must be greater than 100")

                    if experiment["sample_size"] > 100000:
                        validation_errors.append("Sample size must be less than 100000")

            except Exception as e:
                print(f"Error parsing JSON response: {str(e)}")
                validation_errors.append(f"Error parsing LLM response: {str(e)}")

            if not validation_errors:
                break

            if validation_errors:
                error_message = "\n".join(validation_errors)
                system_prompt = self.validate_experiments_prompt.format(
                    error_message=error_message,
                    prompt=state["prompt"],
                    previous_response=json.dumps(state["details"], indent=2),
                )

                state["generate_experiments_prompt"] = system_prompt

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=state["prompt"]),
                ]

                response = self.llm.invoke(messages)
                try:
                    # Extract the first code block from the response
                    code_block = re.search(
                        r"```json\n(.*?)\n```", response.content, re.DOTALL
                    )
                    if not code_block:
                        raise ValueError("No JSON code block found in response")

                    parser = JsonOutputParser()
                    details = parser.parse(code_block.group(1))
                    if not details or details == {}:
                        raise ValueError("Empty or invalid JSON response received")
                    state["details"] = details
                    print("Generated corrected details:", json.dumps(details, indent=2))
                except Exception as e:
                    print(f"Error parsing JSON response: {str(e)}")
                    print("Raw response:", response.content)

            iteration += 1

        return state

    def stage2_confirm_experiments(self, state: CampaignState) -> CampaignState:
        """Stage 2: Confirm experiment details"""

        print("Stage 2: Confirm experiment details")

        if not state["details"]:
            raise ValueError("Please run stage 2 first to generate experiment details")

        user_response = input(
            "\nDo you want to continue with the generated experiment details? (yes/describe the changes you want to make): "
        ).lower()

        if user_response in ["yes", "y"]:
            state["is_experiments_confirmed"] = True
            return state

        state["is_experiments_confirmed"] = False

        # Add historical context to the prompt
        historical_context = ""
        if self.history:
            historical_context = f"""
            Previous Experiment Results:
            {json.dumps(self.history, indent=2)}
            
            Consider these insights for new experiments:
            1. Which variable combinations were most effective
            2. What sample sizes yielded significant results
            3. How different durations affected outcomes
            4. What patterns emerged in ROI performance
            """

        system_prompt = f"""You are an expert in experiment design and statistical analysis.
        Based on the following experiment configuration, determine how many experiments can be run and their variable values.
        
        Previous Prompt: {state["generate_experiments_prompt"]}

        Previous LLM response: {json.dumps(state["details"], indent=2)}
        
        User want to make the following changes: {user_response}

        You MUST return only a valid JSON object in this exact format (with codeblock). Do not return anything else.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content="Generate experiment details with proper statistical controls"
            ),
        ]

        response = self.llm.invoke(messages)
        try:
            # Extract the first code block from the response
            code_block = re.search(r"```json\n(.*?)\n```", response.content, re.DOTALL)
            if not code_block:
                raise ValueError("No JSON code block found in response")

            parser = JsonOutputParser()
            details = parser.parse(code_block.group(1))
            if not details or details == {}:
                raise ValueError("Empty or invalid JSON response received")
            state["details"] = details
            print("Generated details:", json.dumps(details, indent=2))
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            print("Raw response:", response.content)
            raise
        return state

    def stage3_print_details(self, state: CampaignState) -> CampaignState:
        """Stage 3: Print experiment campaign details"""
        print("Stage 3: Print experiment campaign details")
        print(state["details"])
        return state

    def create_graph(self) -> Graph:
        """Create the langgraph workflow"""
        workflow = StateGraph(CampaignState)

        # Add nodes
        workflow.add_node("stage1", self.stage1_generate_config)
        workflow.add_node("stage1_validate", self.stage1_validate_config)
        workflow.add_node("confirm_state1", self.confirm_state1)

        # workflow.add_node("stage1_prompt", self.stage1_prompt)

        workflow.add_node("stage2", self.stage2_generate_experiments)
        workflow.add_node("stage2_validate", self.stage2_validate_experiments)
        workflow.add_node("stage2_confirm", self.stage2_confirm_experiments)

        workflow.add_node("stage3", self.stage3_print_details)

        # Add edges
        workflow.add_edge("stage1", "stage1_validate")
        workflow.add_edge("stage1_validate", "confirm_state1")
        workflow.add_conditional_edges(
            source="confirm_state1",
            path=lambda state: (
                "stage2" if state["is_config_confirmed"] else "stage1_validate"
            ),
        )

        workflow.add_edge("stage2", "stage2_validate")
        workflow.add_edge("stage2_validate", "stage2_confirm")
        workflow.add_conditional_edges(
            source="stage2_confirm",
            path=lambda state: (
                "stage3" if state["is_experiments_confirmed"] else "stage2_validate"
            ),
        )

        workflow.set_entry_point("stage1")
        return workflow.compile()
