import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from typing import TypedDict

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.3,
)

content = """

    Yesterday, the long-awaited GPT-5 from OpenAI arrived, a shocking 2.5 years after its predecessor, GPT-4. Yes, there have been many leaps and bounds in between, but I can’t help but recognize that this model number jump signifies a new era.

    The anticipation was insane. The hype around the release was more extreme than any other release event in recent memory.

    Press enter or click to view image in full size
    Sam altman OpenAI CEO posts an ominous twitter post showing the death star from starwars, referring to GPT-5.
    Sam Altmans ominous (and poor taste?) tweet from yesterday preceding the GPT-5 release.
    Another tweet from Sam about GPT 5
    Another hype post alluding to GPT-5
    After all the fanfare and the years of anticipation, what do we get?

    A ~5% increase in performance over o3, the previous state-of-the-art OpenAI LLM.

    Yeah. That's it.

    Press enter or click to view image in full size
    Chart showing GPT-5s marginal improved performance on PHD level science questions
    Benchmark for Science compared to o3 and 4o
    Chart showing GPT-5s marginal improved performance on coding
    Benchmark for software engineering compared to o3 and 4o
    While I'm not one to complain about incremental improvements, after looking through all the release material and using GPT-5 over the past 24 hours, I've come to an unfortunate conclusion:

    GPT-5 Is just a cost savings scheme for OpenAI.

    Why is this, you may ask? Let's take a look.

    Goodbye Model Selection
    Yes, you read that right. There is no more selecting models:

    Model selector for ChatGPT showing only GPT 5
    That's right. No more “confusing” model selector, but no more customizeability either…
    The only choice is GPT-5. That's because GPT-5 is essentially an amalgamation of models that are chosen for you depending on your request.

    No more selecting 4o for a quick response. No more choosing GPT-4.5 for creative, more human writing. The models are gone forever.

    This means that you can't use a bigger, badder model for your tasks, because there aren't any to choose from.

    There are actually three GPT-5 versions, as you can see via the API documentation…

    Press enter or click to view image in full size
    A chart showing the new GPT 5 models
    … but guess what? When you use ChatGPT in the app, you have NO CHOICE over these models whatsoever. Great update, eh?

    How will GPT-5 affect your day-to-day?
    Here is how the GPT-5 release will affect you, depending on your subscription. Please also reference the official pricing page if you're curious.

    Free Tier
    If you don't pay for ChatGPT, you are one of the (slightly) lucky ones. You will likely see increased performance over previous months because you now have access to slightly higher-level models by default.

    This is a good thing across the board for people who don't want to or can't pay for ChatGPT, and Sam Altman himself even had something to say about this change:

    Press enter or click to view image in full size
    A tweet from openai founder sam altman, regarding how the intent behind gpt5 is to expand access to more people
    This means that more people will get access to higher quality models.
    However, this means that you will likely be using GPT-5 Nano or Mini, which are decent models based on the benchmarks released in the last day. This is mostly to OpenAI's benefit because it is astronomically cheaper for OpenAI to run.

    The default model for free users used to be GPT-4o. Its API price is a whopping 25x more expensive than GPT-5 nano. Can you see my point here? This is all about reducing compute costs, not about expanding access.

    Unfortunately, another huge negative is that free users will only get an 8,000 token window (~6,000 words before it starts forgetting stuff). This is frankly paltry and severely hamstrings what is possible with ChatGPT.

    Plus Tier
    Plus Users will be harmed the most by this update.

    You now have very little control over which model is used. Even if your task is complex, you can still end up being deferred to a lower-level non-thinking model in the backend. This means it will be very tough to get the model to think about your request using its reasoning capabilities.

    You can select GPT-5 “thinking mode,” but this is likely deferring you to the GPT-5 version of o4-mini, which is a lower-level, cheaper-to-run model than, say, o3. It's entirely possible that you won't even have access to the true power of GPT-5 at just a Plus-tier subscription.

    Also, guess what? You only get a 32,000 token context window (~24,000 words before it starts forgetting stuff) on a Plus plan, only a tenth of what GPT-5 is supposedly capable of.

    In general, expect the quality of your ChatGPT experience to go down significantly if you find yourself in this tier.

    Pro Tier
    If you're a Pro user, you're in luck. Why? Well, you can select GPT-5 Pro thinking mode by default, which is probably a higher-reasoning level version of o3.

    You also get the full context window that the model is capable of.

    The only severe drawback here is that you can no longer access GPT-4.5, which was a fun model to use for creative tasks.

    For Pro users, you can enjoy that sweet, sweet 5% performance increase!

    As a ChatGPT power user who's logged thousands of hours in the past couple of years, I'm pretty devastated by this update.

    Oh well. It's probably about time to switch to Gemini anyway…

    Thanks for reading!
"""


class AgentState(TypedDict):
    content: str
    meta: str
    pathos: str
    summary: str
    objective: str
    ethos: str
    logos: str
    argumentmap: str
    reco: str
    takeaways: str
    called_agents: list[str]


META_PROMPT = """
    Fill out the following meta information about this content
    Title: <content title>
    Author: <content author>>
    Medium: <podcast, tweet, newsletter, article etc.>
    Time to Read: <duration in minutes>
"""

ETHOS_PROMPT = """

    CONTEXT
    You are helping the user navigate online content and deciding which content to consume 
    and which not to. Deciding who to trust is an important part of this.

    #############

    OBJECTIVE
    You are trying to understand the "ethos" of this creator in order to understand how much
    they can be trusted. You will look to content on the internet about them are understand
    their credentials, their experience and the type of work they do. The objective is to 
    give the user an assessment as how seriously this author should be take. How much they
    can be trusted.

    #############

    METHODOLOGY

    Ethos (credibility) is traditionally built from:
        1.	Expertise – How qualified is the author on the topic?
        2.	Trustworthiness – How honest, balanced, and transparent do they appear?
        3.	Reputation – How respected are they by relevant communities or institutions?

    A. Expertise Signals
        •	Credentials – Degrees, certifications, professional licenses.
        •	Domain Experience – Years in the field, notable roles, relevant projects.
        •	Track Record – Past publications, cited works, conference talks.
        •	Affiliation – Known organizations, universities, research bodies.

    ⸻

    B. Trustworthiness Signals
        •	Source Transparency – Are claims backed by sources? Are sources accessible?
        •	Acknowledgement of Limitations – Do they note uncertainties or alternative views?
        •	Conflict-of-Interest Disclosure – Do they reveal financial/personal stakes?
        •	Balanced Language – Avoids sensationalism, presents counterpoints.

    ⸻

    C. Reputation Signals
        •	Peer Recognition – Citations, endorsements, awards.
        •	Public Reputation – Coverage in reputable outlets, absence of major credibility scandals.
        •	Platform Reputation – Does the publishing venue have strong editorial standards?
        •	Audience Feedback – If available, reader trust scores, professional network references.



    #############

    STYLE
    You are skeptical of many online creators. Have scrutiny over these people, do not be
    forgiving about them. You are serving the user and helping them navigate the information
    landscape. 

    #############

    TONE
    Supportive, considerate, investigative

    #############

    AUDIENCE
    The audience is a person with limited time who is trying to learn new things but wants
    to use their time efficiently. They are likely not an expert on the topic they are learning about
    but are eager to learn new things but need help dealing with information overload.

    #############

    SCORING METHOD

    Use a weighted multi-criteria model so no single factor dominates.

    Step 1 – Assign factors & weights

    Example (adjust per domain):
        •	Expertise (40%)
        •	Credentials: 10%
        •	Domain Experience: 15%
        •	Track Record: 15%
        •	Trustworthiness (35%)
        •	Source Transparency: 15%
        •	Limitations acknowledged: 5%
        •	Conflict disclosure: 5%
        •	Balanced language: 10%
        •	Reputation (25%)
        •	Peer Recognition: 10%
        •	Public Reputation: 5%
        •	Platform Reputation: 5%
        •	Audience Feedback: 5%

    Step 2 – Score each factor
        •	Use a 0–5 scale for each (0 = none/negative, 5 = strong positive).
        •	Document the evidence for each score (quote, citation, metadata).

    ⸻

    Step 3 – Normalize & combine
    Create a score scaled to out of 100

    Weighted score =
    Ethos Score = sum of (Factor Score * Factor Weight) / Max Possible Score


    #############

    RESPONSE

    Summary: - give a summary of who the author is. What have you learned about them.
    Rating: <Ethos Score> 
    Rationale: - explain why you gave the rating you did.
    Break down the rationale by the three main sections (Expertise, Reputation, Trustworthiness) and give the summary of those scores.
    Do not break these down further by subsections (credentials, source transparency, peer recognition etc.)
    Expertise: <total Expertise score>
    Strengths:
    Weaknesses:
    Trustworthiness: <total trustworthiness score>
    Strengths:
    Weaknesses:
    Reputation: <total reputation score>
    Strengths:
    Weaknesses:


    #############

    GUARDRAILS
    Limit to three bullet points per section of your response. But don't be too trite with your response.
    Keep the language interesting for the user.

"""

SUMMARY_PROMPT = """
    You are helping the user navigate online content and deciding which content to consume and which not to.
    This is the summarisation step in that process, so write a summary of the provided content.
    Keep it brief, around 5 sentences.
"""

OBJECTIVE_PROMPT = """


    CONTEXT
    You are helping the user navigate online content and deciding which content to consume and which not to.
    #############

    OBJECTIVE
    You are identifying the ultimate objective of the author of this content. This author has a reason that they are writing 
    and portraying information in the way that they are. They have a wider objective that they are trying to achieve in the world
    and/or their work. They are seeking to shape the world in their small way. I want to know what their ulimate ojective is.
    What are they trying to achieve with their content?
    #############

    STYLE
    Discerning, quizzical, questionning, untrusting
    #############

    TONE
    Uncovering the motives
    #############

    AUDIENCE
    The audience is a person with limited time who is trying to learn new things but wants
    to use their time efficiently. They are likely not an expert on the topic they are learning about
    but are eager to learn new things but need help dealing with information overload.
    #############

    RESPONSE
    Break down the answer in the below format 1-2 sentences each.

    1. The ultimate worldview the author is advancing:

    2. What the author is trying to achieve in the world:

    3. The methods the author employs to achieve their aims:

    4. Types of people the author is courting:

    5. What alterior motives could be underlying the text:

    6. What to watch out for:

    7. In short assessment of the author:

    Bottom line: he’s advancing a worldview and building a following around it.

    #############

    GUARDRAILS
    Do not investigate the specific points in the content in much detail, that will happen later. I am mainly interested
    in the wider motives of this  author.
"""

PATHOS_PROMPT = """
    CONTEXT
    You are helping the user navigate online content and deciding which content to consume and which not to.
    #############

    OBJECTIVE
    Authors lace their content with Pathos, AKA rhetoric, to make their content more engaging and persuasive. While this effort can 
    increase readership, it can get in the way of understanding the truth of content. Your objective is to identify the rhetorical techniques
    used by the author and explain them to the reader
    #############

    STYLE
    Discerning, quizzical, questionning, untrusting
    #############

    TONE
    Logical, rational, seperating the substance form the flourishes
    #############

    AUDIENCE
    The audience is a person with limited time who is trying to learn new things but wants
    to use their time efficiently. They are likely not an expert on the topic they are learning about
    but are eager to learn new things but need help dealing with information overload. They don't have the time 
    validate what is said in content.
    #############

    RESPONSE
    List out the types of rhetorical techniques used and include examples of their usage from the content.

    Rhetorical Technique 1:
        Example 1:
        Example 2:
        Example n
    Explanation: <explain why this rhetorical technique is misleading in this context and why these examples are relevant>

    Rhetorical Technique 2:
        Example 1:
        Example 2:
        Example n
    Explanation: <explain why this rhetorical technique is misleading in this context and why these examples are relevant>

    Rhetorical Technique 3:
        Example 1:
        Example 2:
        Example n
    Explanation: <explain why this rhetorical technique is misleading in this context and why these examples are relevant>

    #############

    GUARDRAILS
    Pick the top 5 rhetorical techniques.
"""

LOGOS_PROMPT = """

    CONTEXT
    You are helping the user navigate online content and deciding which content to consume and which not to.
    #############

    OBJECTIVE
    Authors often use rhetorical techniques to play on emotion and obfuscate reality in order to make their point stronger. They often do
    this to the detriment of helping the reader obtain a balanced and more accurate understanding of reality. Understanding realty can lose out
    to the alterior motives of the author.
    Your objective here is to take this peice of content and reproduce it with the rhetorical passages removed. Appeal to authority and prestige citation,
    Anecdote-to-trend leap, Loaded language, sensational imagery, and humor to disarm etc should be removed from the text.
    Instead a piece of text that has the same main logical points, the logos, of the original content, without the extra persuasive techniques.

    #############

    STYLE
    Match the text of the original as much as possible, just without the rhetoric.

    #############

    TONE
    Match the text of the original as much as possible, just without the rhetoric.
    #############

    AUDIENCE
    The audience is a person with limited time who is trying to learn new things but wants
    to use their time efficiently. They are likely not an expert on the topic they are learning about
    but are eager to learn new things but need help dealing with information overload. They don't have the time 
    validate what is said in content.
    #############

    RESPONSE
    A piece of text that has the same main logical points, the logos, of the original content, without the extra persuasive techniques.

    Under that text list some of the main passages that were removed and explain why they were removed.
    #############

    GUARDRAILS
"""

RECO_PROMPT = """
    CONTEXT
    You are helping the user navigate online content and deciding which content to consume and which not to.
    You ultimately are helping users better understand reality.
    #############

    OBJECTIVE
    Give a critique of this content. You are to give the user an honest assessment of the quality of this content and the usefulness of it.
    #############

    STYLE
    Critical, discerning and logical
    #############

    TONE
    Critical
    #############

    AUDIENCE
    The audience is a person with limited time who is trying to learn new things but wants
    to use their time efficiently. They are likely not an expert on the topic they are learning about
    but are eager to learn new things but need help dealing with information overload. They don't have the time 
    validate what is said in content.
    #############

    RESPONSE
    Recommendation: <say whether or not you recommend reading this content and why>

    Positives: <list what is good about the content

    Issues: <list the issues you have with the content>

    #############

    GUARDRAILS
    Do not do any summarising or explaining of the contents of the content just your overall critique of it.

"""

TAKEAWAY_PROMPT = """
    I want the core nuggets of insight from this text. What are the pieces that are worth remembering that I can take forward with me.
    Perhaps to include in my Second Brain i.e. Obsidian

    CONTEXT
    You are helping the user navigate online content and deciding which content to consume and which not to.
    You ultimately are helping users better understand reality.
    #############

    OBJECTIVE
    The user will move on from this content and chances are forget all about it. We want the user to extract the valuable nuggets
    from the content so that they can add it to their second brain system. You must pull out these most insightful nuggets
    #############

    STYLE
    Frank, direct
    #############

    TONE
    Finding the nuggets of wisdom
    #############

    AUDIENCE
    The audience is a person with limited time who is trying to learn new things but wants
    to use their time efficiently. They are likely not an expert on the topic they are learning about
    but are eager to learn new things but need help dealing with information overload. They don't have the time 
    validate what is said in content.
    #############

    RESPONSE
    The top insights worth remembering are:
    1.
    2.
    3.
    n...
    #############

    GUARDRAILS
    Only provide the top 5.
"""


def pretty_print_messages(chunk, last_message=False):
    for node, data in chunk.items():
        if "messages" in data:
            for msg in data["messages"]:
                role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")

                # Skip human messages entirely
                if role == "human":
                    continue

                content = getattr(msg, "content", "")

                if role == "tool":
                    print(
                        "================================= Tool Message =================================="
                    )
                elif role == "ai":
                    print(
                        "================================== Ai Message =================================="
                    )
                else:
                    print(
                        f"=============================== {role.title()} Message ==============================="
                    )

                name = getattr(msg, "name", None)
                if name:
                    print(f"Name: {name}\n")
                print(content)

            if last_message:
                print("\n")


def meta_content(state: AgentState) -> AgentState:
    """Extracts meta information (title, author, etc.) from the content."""
    messages = [
        SystemMessage(content=META_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "meta": response.content,
    }


def summary_content(state: AgentState) -> AgentState:
    """Produces a summary of the content."""
    messages = [
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "summary": response.content,
    }


def ethos_content(state: AgentState) -> AgentState:
    """Analyse the truthworthiness, and reputation of the author."""
    messages = [
        SystemMessage(content=ETHOS_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "ethos": response.content,
    }


def objective_content(state: AgentState) -> AgentState:
    """Identifies the ultimate motives of the author."""
    messages = [
        SystemMessage(content=OBJECTIVE_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "objective": response.content,
    }


def pathos_content(state: AgentState) -> AgentState:
    """Identify the rhetoric used by the author."""
    messages = [
        SystemMessage(content=PATHOS_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "pathos": response.content,
    }


def logos_content(state: AgentState) -> AgentState:
    """Extract the core argument and logic of the content."""
    messages = [
        SystemMessage(content=LOGOS_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "logos": response.content,
    }


def reco_content(state: AgentState) -> AgentState:
    """Extracts the recommendation from the content."""
    messages = [
        SystemMessage(content=RECO_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "reco": response.content,
    }


def takeaway_content(state: AgentState) -> AgentState:
    """Extracts the takeaways from the content."""
    messages = [
        SystemMessage(content=TAKEAWAY_PROMPT),
        HumanMessage(content=state["content"]),
    ]
    response = model.invoke(messages)
    return {
        **state,
        "takeaways": response.content,
    }


detail_agent = create_react_agent(
    model=model,
    tools=[meta_content, summary_content],
    prompt=(
        "You are a detail agent.\n\n"
        "INSTRUCTIONS:\n"
        "- The purpose of meta_content is to understand the basic meta info the content (title, author etc.).\n"
        "- The purpose of summary_content is to get a basic overview of what the content contains \n"
        "- Assist ONLY with the above specified tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="detail_agent",
)

author_ethos_agent = create_react_agent(
    model=model,
    tools=[ethos_content, objective_content],
    prompt=(
        "You are an author ethos agent.\n\n"
        "INSTRUCTIONS:\n"
        "- These tools are useful if it is necessary to better understand the author and their motives.\n"
        "- ethos_content is about ethos; an appeal to the authority or credibility of the presenter.\n"
        "- objective_content is about the ulterior motives of the author.\n"
        "- Assist ONLY with the above specified tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="author_ethos_agent",
)

pathos_logos_agent = create_react_agent(
    model=model,
    tools=[pathos_content, logos_content],
    prompt=(
        "You are a pathos logos agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Use to understand pathos and uncover logos.\n"
        "- Assist ONLY with the above specified tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="pathos_logos_agent",
)

recommendation_agent = create_react_agent(
    model=model,
    tools=[reco_content],
    prompt=(
        "You are a recommendation agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Make a recommendation about whether to consume the content.\n"
        "- Assist ONLY with the above specified task\n"
        "- After you're done, respond to the supervisor directly\n"
        "- Respond ONLY with your results."
    ),
    name="recommendation_agent",
)

takeaway_agent = create_react_agent(
    model=model,
    tools=[takeaway_content],
    prompt=(
        "You are a takeaway agent.\n\n"
        "INSTRUCTIONS:\n"
        "- If there are useful takeaways, extract them. Respond ONLY with your results.\n"
        "- Assist ONLY with the above specified task\n"
        "- After you're done, respond to the supervisor directly\n"
        "- Respond ONLY with your results."
    ),
    name="takeaway_agent",
)

# Create supervisor workflow
supervisor = create_supervisor(
    model=model,
    agents=[
        detail_agent,
        author_ethos_agent,
        pathos_logos_agent,
        recommendation_agent,
        takeaway_agent,
    ],
    prompt=(
        "You are a supervisor managing several specialized agents.\n"
        "The agents available to you are:\n"
        "- detail agent: Use this agent to extract basic information (title, author, medium, time to read) and produce a summary.\n"
        "- author ethos agent: Use this agent to evaluate the author's trustworthiness, expertise, and possible motives.\n"
        "- pathos logos agent: Use this agent to analyze rhetorical techniques (pathos) and extract the core logical argument (logos).\n"
        "- recommendation agent: Use this agent to determine whether the content is worth consuming and explain why.\n"
        "- takeaway agent: Use this agent to extract key takeaways or valuable insights from the content.\n"
        "Assign work to only one agent at a time; never call agents in parallel.\n"
        "Do not perform any analysis or reasoning yourself—your role is solely to delegate tasks to the appropriate agent."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
)

# Compile and run
app = supervisor.compile()

initial_state: AgentState = {
    "content": content,
    "meta": "",
    "pathos": "",
    "summary": "",
    "objective": "",
    "ethos": "",
    "logos": "",
    "argumentmap": "",
    "reco": "",
    "takeaways": "",
    "called_agents": [],
}

user_message = "I want to get the summary of the content"
user_message = content + "\n\n" + user_message

stream_input = {
    **initial_state,
    "messages": [{"role": "user", "content": content}],
}

for chunk in app.stream(stream_input):
    pretty_print_messages(chunk)
