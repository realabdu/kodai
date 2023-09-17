# Kodai  : open-source research
to help programmers write better software.


## problem.

When new engineers join a team, they are expected to contribute to the codebase and build features. However, understanding the company's product features, business model, solution architecture, and codebase can be a challenge. Technical documentation can help, but it takes time to create and update, and it may not always be accessible or user-friendly.

## solution

Kodai addresses these challenges by providing an automated documentation solution that is accessed through an AI chat service. The solution consists of two phases:

### **Phase 1: Embedding Current Technical Documentation**

In this phase, Kodai embeds the current technical documentation into the AI chat service. This allows new engineers to easily access the information they need through a conversational interface.

### **Phase 2: Embedding Docs Generated from Codebase**

In this phase, Kodai generates documentation automatically from the codebase and embeds it into the AI chat service. This ensures that the documentation is always up-to-date and reduces the need for manual updates.![My First Board (1)](https://github.com/realabdu/kodai/assets/38006885/f9d2fa8e-bb9e-460a-8b42-c1d4b6796057)

## how can Kodai help your team ?

Kodai is free and open source, and can be self-hosted on your servers. To get started, simply provide the necessary tokens for the OpenAI embedding model `text-embedding-ada-002`  and replicate `meta/llama-2-13b-chat` model. we are working on providing a local LLM option in the future, utilizing edge inference technology.
