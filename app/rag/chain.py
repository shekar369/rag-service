"""LCEL RAG chain: condense question -> hybrid retrieve -> grounded answer.

Output dict: {question, history, standalone, docs, answer}
Every step is a named runnable, so .astream() yields `docs` once and then
`answer` token chunks, which is what the SSE endpoint relies on.
"""
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a standalone search query, "
            "resolving pronouns using the chat history. Return only the query.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer strictly from the provided context.\n"
            "- If the context does not contain the answer, say \"I don't know\" "
            "and state what is missing.\n"
            "- Cite sources inline as [1], [2] matching the context numbering, placed at the "
            "end of the sentence or bullet they support (never alone on a line).\n"
            "- Format in Markdown: a one-line direct answer first, then short bullets or short "
            "paragraphs. Use **bold** for key terms. Use a table only for side-by-side "
            "comparisons, and put citations in a 'Source' column.\n"
            "- Be concise. Never invent facts.",
        ),
        MessagesPlaceholder("history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        loc = d.metadata.get("source", "?")
        if "page" in d.metadata:
            loc += f' p.{int(d.metadata["page"]) + 1}'
        parts.append(f"[{i}] ({loc})\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain(llm, retriever):
    standalone = RunnableBranch(
        (lambda x: not x["history"], itemgetter("question")),  # first turn: no rewrite
        CONDENSE_PROMPT | llm | StrOutputParser(),
    ).with_config(run_name="standalone_question")

    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    ).with_config(run_name="generate_answer")

    return (
        RunnablePassthrough.assign(standalone=standalone)
        .assign(docs=itemgetter("standalone") | retriever.runnable)
        .assign(answer=answer)
    ).with_config(run_name="rag_chain")