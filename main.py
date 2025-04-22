import argparse
from .generator import CampaignGenerator
from .analyser import CampaignAnalyzer

user_prompt = """
I have 10000 users per month. I want to run a campaign to increase the deposit amount.

I want to maximize the Cumulative deposit amount / No of visitors on wallet screen
I want to run the following types of experiments:
    1. Offer - 
        1. Minimum deposit
        2. Reward amount
    2. default payment amount
    3. suggested amount widgets
Use IDR currency
"""


def main():
    parser = argparse.ArgumentParser(description="Campaign Generator Experiment System")
    parser.add_argument(
        "--model",
        choices=["ollama", "anthropic", "openai", "gemini"],
        default="anthropic",
    )
    parser.add_argument("--model-version", default="claude-3-5-haiku-20241022")
    args = parser.parse_args()

    generator = CampaignGenerator(
        model_name=args.model, model_version=args.model_version
    )
    graph = generator.create_graph()

    # Initial state
    initial_state = {
        "current_stage": 1,
        "config": None,
        "details": None,
        "results": [],
        "history": [],
        "prompt": user_prompt,
    }

    print(f"\nRunning experiment cycle")
    result = graph.invoke(initial_state)
    campaign_state = result

    analyzer = CampaignAnalyzer(
        config=campaign_state.get("config", None),
        details=campaign_state.get("details", None),
        model_name=args.model,
        model_version=args.model_version,
    )
    analyzer_state = {
        "current_stage": 3,
        "config": campaign_state.get("config", None),
        "details": campaign_state.get("details", None),
        "results": campaign_state.get("results", []),
        "history": campaign_state.get("history", []),
    }

    print(f"\nRunning analysis cycle")
    analyzer_graph = analyzer.create_graph()
    analyzer_result = analyzer_graph.invoke(analyzer_state)

    # Ask user if they want to run another cycle
    while True:
        response = input(
            "\nDo you want to run another experiment cycle? (yes/no): "
        ).lower()
        if response not in ["yes", "y"]:
            break

        # Use the analysis from the previous cycle as the prompt for the next cycle
        new_state = {
            "current_stage": 3,
            "config": None,
            "details": None,
            "results": [],
            "history": analyzer_result["history"],
            "prompt": analyzer_result["prompt"],
        }

        print(f"\nRunning new experiment cycle")
        result = analyzer_graph.invoke(new_state)
        initial_state = result


if __name__ == "__main__":
    main()
