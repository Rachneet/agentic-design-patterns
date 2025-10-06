from typing import TypedDict, List, Optional


# 1. Define the state structure for the sales report generation process
class SalesReportState(TypedDict):
    request: str
    raw_data: Optional[dict]
    processed_data: Optional[dict]
    chart_config: Optional[dict]
    report: Optional[str]
    errors: List[str]
    next_action: str


# 2. Node functions for each agent
def data_collector_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: collect raw data based on request
    # Update state with raw_data and set next_action
    return state

def data_processor_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: process raw_data and update processed_data
    # Set next_action to next step
    return state

def chart_generator_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: create chart configuration from processed_data
    # Update chart_config and set next_action
    return state

def report_generator_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: generate textual report using processed_data
    # Update report and set next_action to complete
    return state

def error_handler_agent(state: SalesReportState) -> SalesReportState:
    # Placeholder: handle errors, prepare error messages in report
    # Set next_action to complete
    return state


# 3. Routing function
def route_next_step(state: SalesReportState) -> str:
    routing = {
        "collect": "data_collector",
        "process": "data_processor",
        "visualize": "chart_generator",
        "report": "report_generator",
        "error": "error_handler",
        "complete": "END"
    }
    return routing.get(state.get("next_action", "collect"), "END")


# 4. Workflow execution function
from langgraph.graph import StateGraph, END

def create_sales_report_workflow():
    workflow = StateGraph(SalesReportState)

    workflow.add_node("data_collector", data_collector_agent)
    workflow.add_node("data_processor", data_processor_agent)
    workflow.add_node("chart_generator", chart_generator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("error_handler", error_handler_agent)

    workflow.add_conditional_edges(
        "data_collector", 
        route_next_step, {
            "data_processor": "data_processor",
            "error_handler": "error_handler",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "data_processor", 
        route_next_step, {
            "chart_generator": "chart_generator",
            "error_handler": "error_handler",
            END: END
    })
    workflow.add_conditional_edges(
        "chart_generator", 
        route_next_step, {
            "report_generator": "report_generator",
            "error_handler": "error_handler",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "report_generator", 
        route_next_step, {
            "error_handler": "error_handler",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "error_handler", 
        route_next_step, 
        {
            END: END
        }
    )

    workflow.set_entry_point("data_collector")
    return workflow.compile()


# 5. Run the workflow
def run_sales_report_workflow():
    app = create_sales_report_workflow()
    initial_state = SalesReportState(
        request="Q1-Q2 2024 Sales Analysis",
        raw_data=None,
        processed_data=None,
        chart_config=None,
        report=None,
        errors=[],
        next_action="collect"
    )
    print("Starting workflow...\n")
    final_state = app.invoke(initial_state)
    print("\nWorkflow Complete\n")
    if final_state["errors"]:
        print("Errors:")
        for err in final_state["errors"]:
            print(f"- {err}")
    print("\nFinal Report:\n", final_state["report"])
    return final_state


if __name__ == "__main__":
    run_sales_report_workflow()
