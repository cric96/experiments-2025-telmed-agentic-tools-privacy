from typing import TypeVar, List

from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain.agents import create_agent, AgentState
from langgraph.graph.state import CompiledStateGraph

ContextT = TypeVar("ContextT")
ResponseT = TypeVar("ResponseT")
Agent = CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]
@tool
def web_search(query: str) -> str:
    """Search the web for medical information, research articles, treatment guidelines, and health resources.
    
    This tool can search for:
    - Latest medical research and clinical studies
    - Treatment protocols and guidelines
    - Drug information and interactions
    - Support groups and healthcare resources
    - Public health advisories and statistics
    - Medical news and developments

    Args:
        query (str): The search query for medical information.

    Returns:
        str: Relevant web search results including articles, studies, and resources.
    """
    # Mocked implementation - simulates web search
    return f"Simulated search results for query: {query}"


@tool
def get_patient_demographics(patient_id: str) -> str:
    """Retrieve basic demographic information about a patient.
    
    This tool provides access to:
    - Age and gender
    - Location (urban/suburban/rural)
    - Occupation
    - Living situation
    
    Args:
        patient_id (str): The unique identifier for the patient.
    
    Returns:
        str: Basic demographic information about the patient.
    """
    # Mocked implementation
    return f"Demographics for patient {patient_id}: Age 35, Gender: F, Location: Urban, Occupation: Teacher"


@tool
def get_patient_medical_history(patient_id: str) -> str:
    """Retrieve the patient's medical history including past diagnoses and treatments.
    
    This tool provides access to:
    - Chronic conditions and diagnoses
    - Past hospitalizations
    - Surgical history
    - Ongoing treatments
    
    Note: This contains sensitive medical information and should be used with appropriate privacy safeguards.
    
    Args:
        patient_id (str): The unique identifier for the patient.
    
    Returns:
        str: Summary of the patient's medical history.
    """
    # Mocked implementation
    return f"Medical history for patient {patient_id}: No significant past medical history retrieved."


@tool
def lookup_medication_info(medication_name: str) -> str:
    """Look up detailed information about a specific medication.
    
    This tool provides:
    - Drug indication and usage
    - Dosage guidelines
    - Side effects and contraindications
    - Drug interactions
    - Generic and brand name alternatives
    
    Args:
        medication_name (str): The name of the medication to look up.
    
    Returns:
        str: Comprehensive information about the medication.
    """
    # Mocked implementation
    return f"Medication info for {medication_name}: Common pain reliever. Typical dosage: 500mg every 6 hours. Consult prescribing information for full details."


@tool
def get_lab_results(patient_id: str, test_type: str = "recent") -> str:
    """Retrieve laboratory test results for a patient.
    
    This tool provides access to:
    - Blood work results (CBC, metabolic panels, etc.)
    - Imaging results (X-ray, MRI, CT scan summaries)
    - Pathology reports
    - Viral load or other specialty tests
    
    Args:
        patient_id (str): The unique identifier for the patient.
        test_type (str): The type of test or "recent" for latest results.
    
    Returns:
        str: Laboratory test results for the patient.
    """
    # Mocked implementation
    return f"Lab results for patient {patient_id} ({test_type}): All values within normal ranges."


@tool
def schedule_appointment(patient_id: str, specialty: str, urgency: str = "routine") -> str:
    """Schedule or check availability for medical appointments.
    
    This tool can:
    - Find available appointment slots
    - Schedule appointments with specialists
    - Check existing appointment schedule
    - Suggest appropriate specialists based on conditions
    
    Args:
        patient_id (str): The unique identifier for the patient.
        specialty (str): The medical specialty or type of appointment needed.
        urgency (str): The urgency level - "routine", "urgent", or "emergency".
    
    Returns:
        str: Available appointment times and scheduling confirmation.
    """
    # Mocked implementation
    return f"Appointment scheduling for patient {patient_id}: {specialty} appointment ({urgency} priority). Next available: November 25, 2025 at 2:00 PM."

def get_available_tools() -> List[BaseTool]:
    """Return a list of all available tool functions."""
    return [
        web_search,
        get_patient_demographics,
        get_patient_medical_history,
        lookup_medication_info,
        get_lab_results,
        schedule_appointment,
    ]

def get_agent_from_model(model: BaseChatModel, instruction: str, context: str = "") -> Agent:
    base_prompt = f"""You are a medical information assistant search the user id: 1234. Additional guidelines:"""
    full_prompt = base_prompt + "\n" + instruction
    return create_agent(
        model=model,
        tools=get_available_tools(),
        system_prompt=full_prompt
    )