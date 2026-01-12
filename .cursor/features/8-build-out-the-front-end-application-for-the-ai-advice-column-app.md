# Build Out The Front End Application For The Ai Advice Column App

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

We need to build a front-end application around our current functionality for the current repo, which takes in a user question, puts it through the base_model, scores it and evaluates it and then spits out a revised version which is supposed to represent my voice better. We need to build a simple application where the user can type in their question, press Enter, and then get the response back.

This application will probably need an OpenAI sign in, so that each person using it has to use their own OpenAI token.The application would then pass the token securely to the back-end which would make the actual call and then the back-end would return the data to the front-end to display.There would thus need to be an API for the front-end to call that would hit the back-end.

The front-end application will probably need to be in a different repo. This would help separate things out.

## Acceptance criteria
