# AI-Powered Restaurant Recommendation System

## Project context

This project is a restaurant recommendation application inspired by Zomato. It helps a user discover restaurants that match their preferences by combining a real-world restaurant dataset with an LLM. The application should use structured filtering to find relevant options, then use the LLM to rank and explain those options in clear, human-friendly language.

## Problem statement

Restaurant datasets can contain many options, and users often have several criteria in mind at once: where they want to eat, how much they want to spend, the cuisine they prefer, and the minimum rating they will accept. A useful recommendation experience needs to narrow the available choices against these criteria and explain why the final suggestions are a good fit.

The system must provide relevant, understandable recommendations grounded in the available restaurant data. The LLM should help interpret and present the filtered choices; it should not replace the dataset as the source of restaurant facts.

## Objective

Build an application that:

- Collects a user's restaurant preferences.
- Loads and preprocesses a real-world Zomato restaurant dataset.
- Filters the dataset using the user's criteria.
- Uses an LLM to rank suitable restaurants and explain the match.
- Presents the results in a clear, useful format.

## Data source

Use the [Zomato Restaurant Recommendation dataset on Hugging Face](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation). Extract and prepare relevant fields where available, including:

- Restaurant name
- Location
- Cuisine
- Cost
- Rating

Recommendations and displayed details should be based on the records available in this dataset.

## User preferences

The application should support these inputs:

- **Location:** A location available in the restaurant data, selected from a dropdown populated with dataset locations (for example, Delhi or Bangalore).
- **Budget:** A budget category such as low, medium, or high.
- **Cuisine:** A preferred cuisine, such as Italian or Chinese. Cuisine is optional and must not be required to submit a search.
- **Minimum rating:** The lowest acceptable restaurant rating.
- **Additional preferences:** Optional natural-language requirements, such as family-friendly dining or quick service.

## Recommendation workflow

1. Load and preprocess the dataset, selecting and normalizing fields needed for search and display.
2. Collect user preferences through the application interface.
3. Filter the restaurant records against the available structured preferences.
4. Prepare a concise set of relevant records and pass it, together with the user's preferences, to the LLM.
5. Ask the LLM to rank the candidates and explain why each recommendation fits. It may also provide a brief summary of the choices.
6. Display the top recommendations to the user.

## Expected results

Each recommendation should show, when available:

- Restaurant name
- Cuisine
- Rating
- Estimated cost
- An AI-generated explanation connecting the restaurant to the user's preferences

Results should be easy to scan and should clearly distinguish dataset facts from generated explanations.

## LLM integration and operating constraints

Groq is the planned LLM provider. The application should support testing the provider connection with a single call and should avoid unnecessary requests. Pass only the candidate restaurant data and user preference context needed for the current recommendation so that prompts use as few tokens as practical.

The model and rate limits recorded for the project are:

- **Model:** `openai/gpt-oss-120b`
- **Requests:** 30 per minute; 1,000 per day
- **Tokens:** 8,000 per minute; 200,000 per day

Provider credentials belong in a local `.env` file, based on the project's `.env.example` where applicable. Credentials must not be committed to source control or exposed in the user interface.

## Project scope

The project includes dataset ingestion and preparation, preference collection, structured candidate filtering, LLM-assisted ranking and explanations, and a user-facing results experience. Development is intended to proceed in phases documented separately in the architecture, implementation plan, edge-case, and evaluation documents.
