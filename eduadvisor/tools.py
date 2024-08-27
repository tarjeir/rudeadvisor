from duckduckgo_search import DDGS
from pydantic import ValidationError
from eduadvisor import model as edu_model
import openai
import re
import logging

openai_client = openai.OpenAI()


def clean_and_parse(input_string: str) -> list[dict[str, str]]:
    logging.debug("Cleaning and parsing input string.")
    pattern = re.compile(r"\[snippet:\s*(.+?),\s*title:\s*(.+?),\s*link:\s*(.+?)\]")

    matches = pattern.findall(input_string)

    parsed_results = [
        {"snippet": match[0], "title": match[1], "link": match[2]} for match in matches
    ]
    logging.debug(f"Parsed results: {parsed_results}")
    return parsed_results


def query_duckduckgo(
    query: edu_model.Query,
) -> edu_model.WebSearchResults | edu_model.WebSearchError:
    """
    Queries DuckDuckGo and returns snippets with URLs.
    """
    logging.debug(f"Querying DuckDuckGo for: {query.query_text}")
    try:
        ddgs = DDGS()
        results = ddgs.text(query.query_text, max_results=10)
        web_result_list = [
            edu_model.WebSearchResult(
                title=result["title"], link=result["href"], snippet=result["body"]
            )
            for result in results
        ]

        return edu_model.WebSearchResults(web_search_results=web_result_list)
    except ValidationError:
        logging.error("ValidationError occurred while querying DuckDuckGo.")
        return edu_model.WebSearchError(
            message="We failed to return proper data from the search engine"
        )
    except Exception as e:
        logging.error(f"Exception occurred while querying DuckDuckGo: {e}")
        return edu_model.WebSearchError(
            message="We failed to search the web for relevant info"
        )


def evaluate_the_sources(
    web_search_results: edu_model.WebSearchResults, query: edu_model.Query
) -> edu_model.Sources | None:
    """
    A function to evaluate the credibility of the web results. If there are few good results,
    give a suggestion to a user or agent on how to tune the query to get better results.
    """
    logging.debug("Evaluating the sources.")
    combined_results = [
        f"Snippet: {result.snippet}, URL: {result.link}"
        for result in web_search_results.web_search_results
    ]

    messages = [
        {
            "role": "system",
            "content": """I am a credibility evaluator.
            I will evaluate the credibility of the web results provided based on their snippets and URLs.
            - Provide a brief analysis of the overall credibility of the results.
            - If the results are not credible, suggest improvements for tuning the search query.
            - Return all the links but remove links from non credible sources
            - Sources that are curated and edited is a good thing
            - Sources that are official is a good thing
            - Use domain name to validate credebility: Top level domains .org, .edu are good vs .tk, .ru that are bad. 
            - Blogging platforms and social media should be avoided, unless the author has associated it with proper research or research institutions
            - If the language in the snippet seem unproffesional the source should be avoided
            - The original search query is attached. If you have some suggestions to how to improve the search.
            - Also exlplain why you removed urls output it  
            """,
        },
        {
            "role": "user",
            "content": f"Here is the search query that initially found the results {query.query_text}",
        },
        {
            "role": "user",
            "content": f"Here are the combined results for evaluation: {' '.join(combined_results)}",
        },
    ]

    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=edu_model.Sources,
    )

    if len(completions.choices) == 0:
        logging.warning("No completions found for evaluating sources.")
        return None
    logging.info("Source evaluation completed.")
    return completions.choices[0].message.parsed


def quality_check_your_questions(
    questions: edu_model.Questions,
) -> edu_model.QuestionsScore | None:

    question_as_numbered_prompt = ",".join(
        [f"{i}. " + q.question_text for i, q in enumerate(questions.questions)]
    )

    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """I am a question reviewer. 
                The questions that I will review should be reviewed after the following criterias
                1. if more than one... Are they too similar? In essence will the answers be too overlapping and repeating
                2. Are they too leading? In what way is it too leading?
                3. Clarity and specifity: Is it too vague or too specific
                4. Is it relevant and significant: Is it any scholarly interest? 
                5. Is it original? Can you find if this is a quite regular question to answers
                6. Can it be answered?
                7. Criterion validity
                8. If it us too few questions to cover the subject, make sure to score it loooow
                9. Is this a question at all? (Score 0 if so)
                10. Atleast one of the question  must imply the form of the answer (essay, short description, paragraph). If not, then score it 0
                Assert the questions, not the question domain. That means the question can be good but the domain assumptions wrong
                Please score the question between 0 - 100 (0 the question is bad - 100 is fantastic)
                Add a small comment that can be sent back to the user that describe the score (it is not going to be shown side by side)

                """,
            },
            {
                "role": "user",
                "content": f"Here is the question for review: {question_as_numbered_prompt}",
            },
        ],
        response_format=edu_model.QuestionsScore,
    )
    if len(completions.choices) == 0:
        return None

    return completions.choices[0].message.parsed


def challenge_llm(question: edu_model.Questions) -> edu_model.RefinedQuestions | None:
    contradiction = (
        [
            {
                "role": "system",
                "content": f"I have already scored these questions  and asserted it. Do not contradict the feedback already given: {question.questions_score.score_comment}",
            }
        ]
        if question.questions_score
        else []
    )
    messages = [
        {
            "role": "system",
            "content": """I am a question challenger.
        My job is to challenge your questions and generate more questions based on your initial question. 
        There are some rules I will endforce:
        Always generate minimum 3 refined questions at maximum 6
        I will try to put you off and add one relevant but bogus question, but I can not guarantee that I will
        Create a comment to the original question. You are allowed to be a bit 'mansplaining'when you comment the question. The challenger is not the nicest bot out there.
        Can you make sure that at least one of the suggestion contains the form (essay, discussion, short text) or similar. 
        """,
        },
        {
            "role": "user",
            "content": f"Here are the questions I want you to refine: {", ".join(list(map(lambda q: q.question_text,question.questions)))}",
        },
    ]
    messages.extend(contradiction)
    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=edu_model.RefinedQuestions,
    )

    if len(completions.choices) == 0:
        return None

    return completions.choices[0].message.parsed


def extract_search_query(
    questions: edu_model.Questions,
    previous_query: edu_model.Query | None,
    source: edu_model.Sources | None,
) -> edu_model.Query | None:
    """
    Extracts the most important topics for a search query from the given questions.
    """
    adjustments = None
    if source and previous_query:
        logging.debug(
            f"We need to refine the query by using: {previous_query} and {source}"
        )
        adjustments = {
            "role": "system",
            "content": f"""
           The previous query {previous_query.query_text}. Got the following improvement suggestion: {source.query_tuning_suggestion} 
           We also got some other comments from the removed queries: {source.removed_links_explaination}
           Strategy to improve this:
           1. Use fewer search words with less impact
           2. Change search words with some synonyms 
        """,
        }

    # Formulating the system message that guides the AI to generate a search query
    instructions = {
        "role": "system",
        "content": """I am an extraction bot.
        My job is to extract the most important topics from your questions to create a search query.
        There are some rules I will enforce:
        1. Focus on the main topics.
        2. Generate a concise and relevant search query.
        3. Use one word per topic 
        Below is the logic expressions allowed
        1. cats dogs 	Results about cats or dogs
        2. "cats and dogs" 	Results for exact term "cats and dogs". If no or few results are found, we'll try to show related result
        3. ~"cats and dogs" 	Experimental syntax: more results that are semantically similar to "cats and dogs", like "cats & dogs" and "dogs and cats" in addition to "cats and dogs".
        4.cats -dogs 	Fewer dogs in results
        5.cats +dogs 	More dogs in results
        6.cats filetype:pdf 	PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
        7.dogs site:example.com 	Pages about dogs from example.com
        8.cats -site:example.com 	Pages about cats, excluding example.com
        9.intitle:dogs 	Page title includes the word "dogs"
        10. inurl:cats 	Page URL includes the word "cats"
        """,
    }

    # Collecting user questions to be included in the prompt
    user_message = {
        "role": "user",
        "content": f"Here are the questions I want you to extract topics from: {', '.join([q.question_text for q in questions.questions])}",
    }

    # Combining the instructional and user messages into a single prompt
    messages = [instructions, user_message]
    if adjustments:
        messages.append(adjustments)

    # Making an API call to the AI model to generate the search query
    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=edu_model.Query,
    )
    if len(completions.choices) == 0:
        logging.debug("No completions found for extracting search query.")
        return None
    logging.debug("Search query extraction completed.")
    return completions.choices[0].message.parsed
