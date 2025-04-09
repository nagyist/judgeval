from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from .state import GraphState
from .agents import Agents
from .tools import GmailToolsClass
from .structure_outputs import *
from colorama import Fore, Style

# --- Add Judgeval Imports ---
from judgeval.common.tracer import Tracer

# --- Initialize Tracer (or assume it's initialized elsewhere and imported) ---
# If Tracer is initialized in agents.py, we might need to import it differently
# Assuming a global or passed tracer instance for now
# from .agents import judgment # Example if initialized in agents.py
judgment = Tracer(project_name="langgraph-email-agent") # Re-initializing here for simplicity
# --- End Initialization ---


class Nodes():
    def __init__(self):
        self.agents = Agents()
        self.gmail_tools = GmailToolsClass()

    # Load new emails
    @judgment.observe()
    def load_new_emails(self, state: GraphState) -> Dict[str, Any]:
        """Loads new emails from Gmail and updates the state."""
        print(Fore.YELLOW + "Loading new emails...\n" + Style.RESET_ALL)
        recent_emails = self.gmail_tools.fetch_unanswered_emails()
        emails = [Email(**email) for email in recent_emails]
        return {"emails": emails}

    # Check if there are new emails to process
    @judgment.observe()
    def check_new_emails(self, state: GraphState) -> str:
        """Checks if there are new emails to process."""
        if len(state['emails']) == 0:
            print(Fore.RED + "No new emails" + Style.RESET_ALL)
            return "empty"
        else:
            print(Fore.GREEN + "New emails to process" + Style.RESET_ALL)
            return "process"
        
    @judgment.observe()
    def is_email_inbox_empty(self, state: GraphState) -> Dict[str, Any]:
        """Checks if the email inbox is empty."""
        if not state['emails']:
            print(Fore.RED + "Email inbox is empty" + Style.RESET_ALL)
            return {"empty": True}
        else:
            print(Fore.GREEN + "Email inbox is not empty" + Style.RESET_ALL)
            return {"empty": False}

    @judgment.observe()
    def categorize_email(self, state: GraphState) -> Dict[str, Any]:
        """Categorizes the current email."""
        print(Fore.YELLOW + "Checking email category...\n" + Style.RESET_ALL)
        
        # Get the last email
        current_email = state["emails"][-1]
        result = self.agents.categorize_email.invoke({"email": current_email.body})
        print(Fore.MAGENTA + f"Email category: {result.category.value}" + Style.RESET_ALL)
        
        return {
            "email_category": result.category.value,
            "current_email": current_email
        }

    @judgment.observe()
    def route_email_based_on_category(self, state: GraphState) -> str:
        """Routes the email based on its category."""
        print(Fore.YELLOW + "Routing email based on category...\n" + Style.RESET_ALL)
        category = state["email_category"]
        if category == "product_enquiry":
            return "product related"
        elif category == "unrelated":
            return "unrelated"
        else:
            return "not product related"

    @judgment.observe()
    def construct_rag_queries(self, state: GraphState) -> Dict[str, Any]:
        """Constructs RAG queries based on the current email."""
        print(Fore.YELLOW + "Constructing RAG queries...\n" + Style.RESET_ALL)
        email_content = state["current_email"].body
        query_result = self.agents.design_rag_queries.invoke({"email": email_content})
        
        return {"rag_queries": query_result.queries}

    @judgment.observe()
    def retrieve_from_rag(self, state: GraphState) -> Dict[str, Any]:
        """Retrieves documents from RAG based on queries."""
        print(Fore.YELLOW + "Retrieving documents from RAG...\n" + Style.RESET_ALL)
        final_answer = ""
        for query in state["rag_queries"]:
            rag_result = self.agents.generate_rag_answer.invoke(query)
            final_answer += query + "\n" + rag_result + "\n\n"
        
        return {"retrieved_documents": final_answer}

    @judgment.observe()
    def write_draft_email(self, state: GraphState) -> Dict[str, Any]:
        """Writes a draft email based on the current state."""
        print(Fore.YELLOW + "Writing draft email...\n" + Style.RESET_ALL)
        
        # Format input to the writer agent
        inputs = (
            f'# **EMAIL CATEGORY:** {state["email_category"]}\n\n'
            f'# **EMAIL CONTENT:**\n{state["current_email"].body}\n\n'
            f'# **INFORMATION:**\n{state["retrieved_documents"]}' # Empty for feedback or complaint
        )
        
        # Get messages history for current email
        writer_messages = state.get('writer_messages', [])
        
        # Write email
        draft_result = self.agents.email_writer.invoke({
            "email_information": inputs,
            "history": writer_messages
        })
        email = draft_result.email
        trials = state.get('trials', 0) + 1

        # Append writer's draft to the message list
        writer_messages.append(f"**Draft {trials}:**\n{email}")

        return {
            "generated_email": email, 
            "trials": trials,
            "writer_messages": writer_messages
        }

    @judgment.observe()
    def verify_generated_email(self, state: GraphState) -> Dict[str, Any]:
        """Verifies the generated email for correctness and sendability."""
        print(Fore.YELLOW + "Verifying generated email...\n" + Style.RESET_ALL)
        review = self.agents.email_proofreader.invoke({
            "initial_email": state["current_email"].body,
            "generated_email": state["generated_email"],
        })

        writer_messages = state.get('writer_messages', [])
        writer_messages.append(f"**Proofreader Feedback:**\n{review.feedback}")

        return {
            "sendable": review.send,
            "writer_messages": writer_messages
        }

    @judgment.observe()
    def must_rewrite(self, state: GraphState) -> str:
        """Determines if the email needs to be rewritten based on the review and trial count."""
        email_sendable = state["sendable"]
        if email_sendable:
            print(Fore.GREEN + "Email is good, ready to be sent!!!" + Style.RESET_ALL)
            state["emails"].pop()
            state["writer_messages"] = []
            return "send"
        elif state["trials"] >= 3:
            print(Fore.RED + "Email is not good, we reached max trials must stop!!!" + Style.RESET_ALL)
            state["emails"].pop()
            state["writer_messages"] = []
            return "stop"
        else:
            print(Fore.RED + "Email is not good, must rewrite it..." + Style.RESET_ALL)
            return "rewrite"

    @judgment.observe()
    def create_draft_response(self, state: GraphState) -> Dict[str, Any]:
        """Creates a draft response in Gmail."""
        print(Fore.YELLOW + "Creating draft response...\n" + Style.RESET_ALL)
        self.gmail_tools.create_draft_reply(state["current_email"], state["generated_email"])
        
        return {"retrieved_documents": "", "trials": 0}

    @judgment.observe()
    def send_email_response(self, state: GraphState) -> GraphState:
        """Sends the email response directly using Gmail."""
        print(Fore.YELLOW + "Sending email...\n" + Style.RESET_ALL)
        self.gmail_tools.send_reply(state["current_email"], state["generated_email"])
        
        return {"retrieved_documents": "", "trials": 0}
    
    @judgment.observe()
    def skip_unrelated_email(self, state: GraphState) -> Dict[str, Any]:
        """Handles the skipping of unrelated emails."""
        print(Fore.YELLOW + "Skipping unrelated email...\n" + Style.RESET_ALL)
        state["emails"].pop()
        return {"skipped": True}