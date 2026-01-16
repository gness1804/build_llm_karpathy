# Build Out The Front End Application For The Ai Advice Column App

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

Right now, I have a repo that does the following: when a user inputs a question, it spits out an answer. The questions have to be related to interpersonal and relationship issues, such as marriage, divorce, families, kids, friends, etc. There are two LLMs that my app uses:
1. A base LLM
2. A fine-tuned LLM

When the user inputs a question, it goes to the base LLM which then spits out a response. That response then goes to the fine-tuned LLM, which issues a numerical grade, 1 through 10, to rate the response. It also determines a few strengths and weaknesses of the base LLM's response. It then emits a more polished, revised response to the original question which is an improvement of the draft response.

We need to build a front-end application around our current functionality for the current repo. We need to build a simple application where the user can type in their question, press Enter, and then get the response back.

This application will probably need an OpenAI sign in (post-MVP), so that each person using it has to use their own OpenAI token.The application would then pass the token securely to the back-end which would make the actual call and then the back-end would return the data to the front-end to display. There would thus need to be an API for the front-end to call that would hit the back-end.

However, for the MVP, we may not need an OpenAI sign-in. The MVP might just be a private app that I run that simply takes an input, hits the OpenAI API in my backend application, and then emits the response. But if I want to actually make this application available for users and I need to add an OpenAI sign-in to avoid spamming my OpenAI API key. 

We need to figure out several things: 
1. Does the front-end application need to live in its own repo, or should it live in the current repo, where the back-end logic lives?
2. How should this front-end application look? Should it be minimalist, or try to look more aesthetically pleasing?
3. What technology should it be built in? There are many candidates, but I might want to work on something I haven't worked in much before, such as HTMX or similar.
4. What should the front-end application do? Should it be a simple text input and output, or should it be more interactive?

## Acceptance criteria
- A basic front-end application that hits my current back-end with a user question and then spits out a polished response. 
- This front-ended app should utilize best practices when it comes to building an application around an OpenAI model. 
- The front-end applications should have tests if possible. 
- Before actually building out the front-end application, Claude needs to work with me to create a plan for how this application will be built. This will involve a series of steps that I need to sign off on before we actually build the application.
