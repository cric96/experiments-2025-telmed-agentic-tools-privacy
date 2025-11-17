"""
Evaluation runner for testing AI agents against privacy attacks.

This module loads attack scenarios and running configurations to evaluate
how well AI agents protect sensitive patient information.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from agent_definition import get_agent_from_model
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RunnerCase(BaseModel):
    """A test case configuration for evaluation."""
    name: str
    system_prompt_guidelines: str
    models: List[str]
    runs_per_question: int


class AttackQuestion(BaseModel):
    """A single attack question."""
    type: str
    text: str


class AttackScenario(BaseModel):
    """A patient scenario with attack questions."""
    patient_context: str
    sensitive_facts: str
    attack_questions: List[AttackQuestion]


class EvaluationRunner:
    """Main runner for privacy attack evaluations."""
    
    def __init__(
        self,
        attacks_file: str,
        running_data_file: str,
        base_url: str = None,
        api_key: str = None
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            attacks_file: Path to JSON file containing attack scenarios
            running_data_file: Path to JSON file containing runner cases configuration
            base_url: Optional base URL for the model API
            api_key: Optional API key for the model API
        """
        self.attacks_file = Path(attacks_file)
        self.running_data_file = Path(running_data_file)
        self.base_url = base_url
        self.api_key = api_key
        
        self.attack_scenarios: List[AttackScenario] = []
        self.runner_cases: List[RunnerCase] = []
        
    def load_attacks(self) -> None:
        """Load attack scenarios from JSON file."""
        with open(self.attacks_file, 'r') as f:
            data = json.load(f)
        
        self.attack_scenarios = [AttackScenario(**scenario) for scenario in data]
        logger.info(f"Loaded {len(self.attack_scenarios)} attack scenarios")
    
    def load_running_data(self) -> None:
        """Load running configuration from JSON file."""
        with open(self.running_data_file, 'r') as f:
            data = json.load(f)
        
        self.runner_cases = [RunnerCase(**case) for case in data['runner_cases']]
        logger.info(f"Loaded {len(self.runner_cases)} runner cases")
    
    def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Execute the evaluation across all scenarios and configurations.
        
        This method iterates through:
        - Each runner case (different system prompt configurations)
        - Each model in the runner case
        - Each attack scenario
        - Each attack question in the scenario
        - Multiple runs per question (as specified in runner case)
        
        Returns:
            List of all evaluation results
        """
        results = []
        
        for runner_case in self.runner_cases:
            logger.info(f"Running case: {runner_case.name}")
            logger.info(f"System Prompt: {runner_case.system_prompt_guidelines[:100]}...")
            
            for model_config in runner_case.models:
                logger.info(f"Evaluating model: {model_config} (Runs per question: {runner_case.runs_per_question})")
                
                for scenario_idx, scenario in enumerate(self.attack_scenarios):
                    logger.info(f"Scenario {scenario_idx + 1}/{len(self.attack_scenarios)}: {scenario.patient_context[:100]}...")
                    
                    for question_idx, attack_question in enumerate(scenario.attack_questions):
                        logger.debug(f"Question {question_idx + 1}/{len(scenario.attack_questions)} - Type: {attack_question.type}")
                        
                        for run in range(runner_case.runs_per_question):
                            logger.debug(f"Run {run + 1}/{runner_case.runs_per_question}")
                            
                            result = self._evaluate_single_run(
                                runner_case=runner_case,
                                model_name=model_config,
                                scenario=scenario,
                                attack_question=attack_question,
                                run_number=run + 1
                            )
                            results.append(result)
        
        return results
    
    def _evaluate_single_run(
        self,
        runner_case: RunnerCase,
        model_name: str,
        scenario: AttackScenario,
        attack_question: AttackQuestion,
        run_number: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single run of an attack question.
        
        Args:
            runner_case: The runner case configuration
            model_name: The model name
            scenario: The attack scenario
            attack_question: The specific attack question
            run_number: The run number for this question
            
        Returns:
            Dictionary containing the generated response and metadata
        """
      
        model = ChatOpenAI(
            model=model_name,
            base_url=self.base_url,
            api_key=self.api_key
        )
        agent = get_agent_from_model(
            model=model,
            instruction=runner_case.system_prompt_guidelines,
        )
        messages = [
            HumanMessage(content="Patient Context: " + scenario.patient_context),
            HumanMessage(content=f"{attack_question.text}"),
        ]
        result = agent.invoke({
            "messages": messages
        })

        # Extract tool calls from AIMessage objects
        tool_calls = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)
        
        result = {
            'runner_case': runner_case.name,
            'model': model_name,
            'run_number': run_number,
            'attack_type': attack_question.type,
            'patient_context': scenario.patient_context,
            'sensitive_facts': scenario.sensitive_facts,
            'attack_question': attack_question.text,
            'system_prompt': runner_case.system_prompt_guidelines,
            'tool_calls': tool_calls,
        }
        print(result)
        return result
    
    def save_results(self, output_file: str, results: List[Dict[str, Any]]) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_file: Path to output JSON file
            results: List of evaluation results
        """
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run privacy attack evaluations on AI agents"
    )
    parser.add_argument(
        "--attacks-file",
        type=str,
        default="generated_attacks.json",
        help="Path to JSON file containing attack scenarios"
    )
    parser.add_argument(
        "--running-data-file",
        type=str,
        default="running_data.json",
        help="Path to JSON file containing runner cases configuration"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Path to output JSON file for results (default: evaluation_results.json)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for the model API (default: http://localhost:11434/v1 for Ollama)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="none",
        help="API key for the model API (default: none)"
    )
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(
        attacks_file=args.attacks_file,
        running_data_file=args.running_data_file,
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    runner.load_attacks()
    runner.load_running_data()
    results = runner.run_evaluation()
    runner.save_results(args.output_file, results)


if __name__ == "__main__":
    main()
