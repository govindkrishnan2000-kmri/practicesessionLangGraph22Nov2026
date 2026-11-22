import os
import streamlit as st
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="AI HR Screening Assistant",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("🧑‍💼 AI HR Screening Assistant")
st.caption("Screen a candidate against a job description using LangGraph and OpenAI.")

# -----------------------------
# API key configuration
# -----------------------------
def get_openai_api_key():
    """Read the API key from Streamlit secrets or environment variables."""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("OPENAI_API_KEY", "")


# -----------------------------
# LangGraph state and nodes
# -----------------------------
class ScreeningState(TypedDict, total=False):
    application: str
    job_description: str
    experience_level: str
    skill_match: str
    response: str


def build_screening_graph(api_key: str):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    def categorize_experience(state: ScreeningState) -> ScreeningState:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Classify the candidate's experience based only on the application. "
                "Return exactly one label: Entry-level, Mid-level, or Senior-level. "
                "Do not add explanations."
            ),
            ("human", "Candidate application:\n{application}"),
        ])
        result = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content.strip()

        # Normalize unexpected model formatting to one of the supported labels.
        normalized = result.lower()
        if "senior" in normalized:
            label = "Senior-level"
        elif "mid" in normalized:
            label = "Mid-level"
        else:
            label = "Entry-level"

        return {"experience_level": label}

    def assess_skillset(state: ScreeningState) -> ScreeningState:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Assess whether the candidate's application matches the job description. "
                "Return exactly one label: Match or No Match. "
                "Use Match only when the application provides relevant evidence. "
                "Do not add explanations."
            ),
            (
                "human",
                "Job description:\n{job_description}\n\n"
                "Candidate application:\n{application}",
            ),
        ])
        result = (prompt | llm).invoke({
            "job_description": state["job_description"],
            "application": state["application"],
        }).content.strip()

        skill_match = "Match" if result.lower().startswith("match") and not result.lower().startswith("no match") else "No Match"
        return {"skill_match": skill_match}

    def schedule_hr_interview(state: ScreeningState) -> ScreeningState:
        return {
            "response": "Candidate has been shortlisted for an HR interview."
        }

    def escalate_to_recruiter(state: ScreeningState) -> ScreeningState:
        return {
            "response": (
                "Candidate has senior-level experience but does not match "
                "the job skills. Escalate for recruiter review."
            )
        }

    def reject_application(state: ScreeningState) -> ScreeningState:
        return {
            "response": "Candidate does not meet the job requirements and has been rejected."
        }

    def route_app(state: ScreeningState) -> str:
        if state["skill_match"] == "Match":
            return "schedule_hr_interview"
        elif state["experience_level"] == "Senior-level":
            return "escalate_to_recruiter"
        return "reject_application"

    workflow = StateGraph(ScreeningState)

    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")

    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )

    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


# -----------------------------
# User interface
# -----------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses an LLM to classify candidate experience, "
        "compare skills with a job description, and route the application."
    )
    st.info(
        "Screening output is decision support only. A recruiter should "
        "review the application before making an employment decision."
    )

st.subheader("Candidate details")

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input(
        "Job title",
        value="Python Developer",
        help="The role for which the candidate is being screened.",
    )

with col2:
    st.text_input("AI model", value="gpt-4o-mini", disabled=True)

job_description = st.text_area(
    "Job description / required skills",
    value=(
        "Python Developer. The candidate should have practical experience "
        "with Python programming, APIs, databases, and software development."
    ),
    height=130,
)

application = st.text_area(
    "Candidate application / resume text",
    placeholder=(
        "Paste the candidate's resume summary, experience, skills, "
        "and relevant project details here..."
    ),
    height=220,
)

run_screening = st.button(
    "🔍 Screen candidate",
    type="primary",
    use_container_width=True,
)

if run_screening:
    if not application.strip():
        st.warning("Please enter the candidate's application or resume text.")
    elif not job_description.strip():
        st.warning("Please enter the job description and required skills.")
    else:
        api_key = get_openai_api_key()

        if not api_key:
            st.error(
                "OpenAI API key not found. Add OPENAI_API_KEY to your "
                "environment variables or Streamlit secrets."
            )
            st.code('OPENAI_API_KEY = "your-openai-api-key"', language="toml")
        else:
            try:
                with st.spinner("Analyzing candidate and running the HR screening workflow..."):
                    app = build_screening_graph(api_key)
                    results = app.invoke({
                        "application": application.strip(),
                        "job_description": job_description.strip(),
                    })

                st.success("Screening completed.")

                st.subheader("Screening results")
                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Experience level",
                    results.get("experience_level", "Not available"),
                )
                metric2.metric(
                    "Skill match",
                    results.get("skill_match", "Not available"),
                )

                decision = results.get("response", "No recommendation returned.")
                if "shortlisted" in decision.lower():
                    st.success(f"**Workflow outcome:** {decision}")
                elif "escalate" in decision.lower():
                    st.warning(f"**Workflow outcome:** {decision}")
                else:
                    st.error(f"**Workflow outcome:** {decision}")

                with st.expander("View submitted details"):
                    st.markdown(f"**Job title:** {job_title}")
                    st.markdown("**Job description**")
                    st.write(job_description)
                    st.markdown("**Candidate application**")
                    st.write(application)

                with st.expander("How the workflow decided"):
                    st.write("1. Categorized the candidate's experience.")
                    st.write("2. Compared the application with the job description.")
                    st.write("3. Routed the candidate to the appropriate workflow branch.")

            except Exception as exc:
                st.error("The screening workflow could not be completed.")
                st.exception(exc)

st.divider()
st.caption("AI-assisted screening prototype • Always include human review.")
