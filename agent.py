from datetime import datetime
import requests
from openai import OpenAI
from pydantic import Field, BaseModel
from typing import Optional, Literal, List
import wikipedia

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

class RequestType(BaseModel):
    request_type: Literal["new_event", "modify_event", "cancel_event", "list_events", "wikipedia_query", "other"] = Field(description="Type of request handled by the agent.")
    confidence_score: float = Field(description="Confidence score between 0 an 1.")
    description: str = Field(description="Cleaned description of the request.")

class NewEventDetails(BaseModel):
    name: str = Field(description="Name of the calendar event")
    date: str = Field(description="Date of the calendar event")
    duration_minutes: int = Field(description="How long will it be")
    participants: list[str] = Field(description="The list of participants invited")

class Change(BaseModel):
    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")

class ModifyEventDetails(BaseModel):
    event_identifier: str = Field(description="Description to identify the existing event.")
    changes: list[Change] = Field(description="List of changes to apply")
    participants_to_add: list[str] = Field(description="Participants to be added")
    participants_to_remove: list[str] = Field(description="Participants to be removed")

class ListEventsDetails(BaseModel):
    date_range_start: Optional[str] = Field(description="Start of the date range for listing events")
    date_range_end: Optional[str] = Field(description="End of the date range for listing events")

class CancelEventDetails(BaseModel):
    event_identifier: str = Field(description="Description to identify the existing event.")

class CalendarResponse(BaseModel):
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if any")

class WikipediaResponse(BaseModel):
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User friendly response message")
    summary: Optional[str] = Field(description="Wikipedia summary of the query")

def route_calendar_request(user_input: str) -> RequestType:
    completion = client.beta.chat.completions.parse(
     model="llama3.1",
     messages=[
       {
       "role": "system",
       "content": "Determine the type of request (new_event, modify_event, cancel_event, list_events, wikipedia_query, other)"
       },
       {
       "role": "user",
       "content": user_input
       }
     ],
     response_format=RequestType
    )

    result = completion.choices[0].message.parsed
    return result

def route_request(user_input: str) -> RequestType:
    completion = client.beta.chat.completions.parse(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": ("Determine the type of request (new_event, modify_event, cancel_event, "
                            "list_events, wikipedia_query, other)")
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        response_format=RequestType
    )

    result = completion.choices[0].message.parsed
    return result

def handle_modify_event(modify_details: ModifyEventDetails) -> CalendarResponse:
    # Assuming modify_details is already parsed and contains the necessary information
    event_identifier = modify_details.event_identifier
    changes = modify_details.changes
    participants_to_add = modify_details.participants_to_add
    participants_to_remove = modify_details.participants_to_remove

    # Implement logic to modify the event based on the details provided
    return CalendarResponse(
        success=True, 
        message=f"Modified Event '{event_identifier}'", 
        calendar_link="calendar://modified"
    )


def handle_list_events(list_details: ListEventsDetails) -> CalendarResponse:
    # Assuming list_details is already parsed and contains the necessary information
    date_range_start = list_details.date_range_start
    date_range_end = list_details.date_range_end

    # Implement logic to list events based on the details provided
    return CalendarResponse(
        success=True,
        message=f"Listing Events from '{date_range_start}' to '{date_range_end}'",
        calendar_link="calendar://list"
    )

def handle_wikipedia_query(query: str) -> WikipediaResponse:
    try:
        summary = wikipedia.summary(query, sentences=3)
        return WikipediaResponse(
            success=True,
            message=f"Wikipedia Summary for '{query}': {summary}",
            summary=summary
        )
    except wikipedia.exceptions.DisambiguationError as e:
        return WikipediaResponse(
            success=False,
            message=f"Disambiguation error for query '{query}'. Please be more specific. Options: {', '.join(e.options)}",
            summary=None
        )
    except wikipedia.exceptions.PageError:
        return WikipediaResponse(
            success=False,
            message=f"No Wikipedia page found for query '{query}'.",
            summary=None
        )

def handle_cancel_event(cancel_details: CancelEventDetails) -> CalendarResponse:
    # Assuming cancel_details is already parsed and contains the necessary information
    event_identifier = cancel_details.event_identifier

    # Implement logic to cancel the event based on the details provided
    return CalendarResponse(
        success=True,
        message=f"Canceled Event '{event_identifier}'",
        calendar_link="calendar://canceled"
    )

def handle_new_event(description: str) -> CalendarResponse:

    completion = client.beta.chat.completions.parse(
      model="llama3.1",
      messages=[
        {
        "role": "system",
        "content": "Extract details for a new calendar event"
        },
        {
        "role": "user",
        "content": description
        }
      ],
      response_format=RequestType
    )

    details = completion.choices[0].message.parsed

    return CalendarResponse(success=True,
     message=f"Created new event '{details.name}', '{details.date}'",
     calendar_link=f"calendar://new?event={details.name}"
    )

def parse_event_details(description: str) -> NewEventDetails:
    today = datetime.today()
    date_context = f"Today is {today.strftime('%A %B %d %Y')}."

    completion = client.beta.chat.completions.parse(
       model="llama3.1",
       messages = [
        {
         "role": "system",
         "content": f"{date_context} Extract detailed event information."
        },
        {
         "role": "user",
         "content": description
        }
       ],
       response_format=RequestType
    )

    result = completion.choices[0].message.parsed
    return result

from typing import Optional, Literal, List

class DecisionEvaluation(BaseModel):
    score: float = Field(description="Evaluation score between 0 and 1 where 1 means highly beneficial and 0 means not beneficial")

class TaskStatus(BaseModel):
    completed: bool = Field(description="Whether the task is completed")
    message: str = Field(description="Message from the agent")

from typing import List, Optional

class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps in the plan")

def evaluate_decision(user_input: str, decision: str) -> DecisionEvaluation:
    """
    Evaluate the decision and return a score indicating if it will benefit the resolution.
    Score is between 0 and 1 where 1 means highly beneficial and 0 means not beneficial.
    """
    completion = client.beta.chat.completions.parse(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": f"Evaluate the decision '{decision}' for the task described in the user input. Return a score between 0 and 1 where 1 means highly beneficial and 0 means not beneficial."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        response_format=DecisionEvaluation
    )

    result = completion.choices[0].message.parsed
    return result

def create_plan(user_input: str) -> Plan:
    completion = client.beta.chat.completions.parse(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": "Create a step-by-step plan to complete the task described in the user input."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        response_format=Plan
    )

    result = completion.choices[0].message.parsed
    return result

def evaluate_task_completion(user_input: str, attempt: int) -> TaskStatus:
    completion = client.beta.chat.completions.parse(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": f"Check if the task described in the user input is completed. Attempt {attempt}."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        response_format=TaskStatus
    )

    result = completion.choices[0].message.parsed
    return result

from typing import Union, List

steps_list: List[str] = []
uncapable_of: List[str] = []
completed_steps: List[str] = []

class CapabilityEvaluation(BaseModel):
    capable_steps: List[str] = Field(description="List of steps the model is capable of handling")
    incapable_steps: List[str] = Field(description="List of steps the model is not capable of handling")

class StepComparisonResponse(BaseModel):
    equivalent: bool = Field(description="Whether the steps are semantically equivalent")

def is_step_already_in_pending_steps(new_step: str, pending_steps: List[str], completed_steps: List[str]) -> StepComparisonResponse:
    """
    Determine if a proposed step is already in the list of pending steps or completed steps by analyzing the semantics.
    """
    for step_list in [pending_steps, completed_steps]:
        for step in step_list:
            completion = client.beta.chat.completions.parse(
                model="llama3.1",
                messages=[
                    {
                        "role": "system",
                        "content": f"Compare the following two tasks and determine if they are semantically equivalent: Task 1: '{step}', Task 2: '{new_step}'. Return 'yes' if they are equivalent, otherwise return 'no'."
                    }
                ],
                response_format=Literal["yes", "no"]
            )
            result = completion.choices[0].message.parsed
            if result == "yes":
                return StepComparisonResponse(equivalent=True)
    return StepComparisonResponse(equivalent=False)

def evaluate_capabilities(steps: List[str]) -> CapabilityEvaluation:
    """
    Evaluate the capabilities of the model for each step in the list.
    Return a list of steps the model can handle and a list it cannot.
    """
    # Placeholder logic to determine capability
    capable_steps = []
    incapable_steps = []

    for step in steps:
        if "unknown" in step.lower() or "confused" in step.lower():
            incapable_steps.append(step)
        else:
            capable_steps.append(step)

    return CapabilityEvaluation(capable_steps=capable_steps, incapable_steps=incapable_steps)

async def agent_loop(user_input: str, max_attempts: int = 15): # -> Optional[Union[CalendarResponse, WikipediaResponse]]:
    global steps_list
    global uncapable_of
    attempt = 1
    while attempt <= max_attempts:
        route_result = route_request(user_input)
        print(route_result)
        
        decision = f"Request type: {route_result.request_type}, Description: {route_result.description}"
        evaluation_result = evaluate_decision(user_input, decision)
        evaluation_score = evaluation_result.score
        print(f"Evaluation score for decision '{decision}': {evaluation_score}")

        if evaluation_result.score < 0.5:
            plan = create_plan(user_input)
            for step in plan.steps:
                if not is_step_already_in_pending_steps(step, steps_list, completed_steps):
                    steps_list.append(step)
            print("Plan:", plan.steps)

            capability_evaluation = evaluate_capabilities(steps_list)
            steps_list = capability_evaluation.capable_steps
            uncapable_of.extend(capability_evaluation.incapable_steps)
            print("Capable Steps:", steps_list)
            print("Incapable Steps:", uncapable_of)

            attempt += 1
            continue

        if route_result.request_type == "new_event":
            response = handle_new_event(route_result.description)
        elif route_result.request_type == "modify_event":
            # Ensure the description is correctly parsed for modify_event
            modify_details = parse_event_details(route_result.description)
            response = handle_modify_event(modify_details)
        elif route_result.request_type == "cancel_event":
            cancel_details = parse_event_details(route_result.description)
            response = handle_cancel_event(cancel_details)
        elif route_result.request_type == "list_events":
            list_details = parse_event_details(route_result.description)
            response = handle_list_events(list_details)

        elif route_result.request_type == "wikipedia_query":
            response = handle_wikipedia_query(route_result.description)
        elif route_result.request_type == "other":  # Handle 'other' request type
            plan = create_plan(user_input)
            steps_list.extend(plan.steps)  # Add new steps to the global list
            print("Plan:", plan.steps)

        task_status = evaluate_task_completion(user_input, attempt)
        print(task_status)

        if task_status.completed and not steps_list:
            steps_list.clear()  # Clear the steps list if the task is completed
            uncapable_of.clear()  # Clear the incapable steps list if the task is completed
            completed_steps.extend(steps_list)  # Move completed steps to the completed_steps list
            return None

        attempt += 1

    return None

if __name__ == "__main__":
    import asyncio

    # Sample user input for testing
    #user_input = "Create a new event named 'Team Meeting' the day in which the independence of USA was declared (check wikipedia) at 3 PM, lasting 60 minutes with participants Alice and Bob. Prior to answering, think step by step of the available tools and make a plan. Reevaluate at each step what should be done"
    user_input = "where does cristiano ronaldo play?"
    #user_input = "how to solve differential equations"

    # Run the agent loop
    response = asyncio.run(agent_loop(user_input))

    # Print the response
    if response:
        print("Response:", response)
    else:
        print("No valid response received.")

