# CodeDecoder
## Inspiration
Learning coding can be daunting. We're sure many of us would have loved to have a coach by our side to explain what each line of code does when we were first faced with a scary-looking block of code (both of us sure did!). This, coupled with the rising number of self-taught coders (85.5% according to a 2019 Stack overflow survey), highlights the importance of real time feedback to 1. ease learners into coding and 2. develop good coding practices from day one. This is why we came up with CodeDecoder, a multilingual deep learning powered educational tool to translate code into natural language.

## What it does
CodeDecoder is currently fluent in four languages: Python, Java, JavaScript and Go. When learners first chance upon a code snippet, they can simply send it to CodeDecoder which will parse the code and return an explanation of what the code does.

CodeDecoder consists of two components: 1. A **web app** and 2. A **Visual Studio Code extension**. 
1. With the web app, users can either copy and paste their code snippet into the text box or upload a file containing the code to receive an explanation of the code. 
2. With the Visual Studio Code extension, users can simply highlight the code snippet, right click and choose the "Decode" button to receive an explanation of the code.

## How we built it
We fine-tuned the state-of-the-art T5 NLP model on code and natural language pairs for four different languages (Python, Java, JavaScript and Go) using PyTorch and the Huggingface transformers library. We then deployed these four transformer models using the Streamlit library for the web app.
The Visual Studio Code extension is coded in **Typescript** and **Javascript**. We used the **FastAPI** library to serve the deep learning models, and then linked the backend to the extension using **axios**.

## Challenges we ran into
Being the first time we used FastAPI to link our deep learning models (coded in Python) with the Visual Studio Code extension (coded in Typescript and Javascript), we faced a lot of challenges trying to get them to talk to each other.

## Accomplishments that we're proud of
We're really proud to have built two components of CodeDecoder -- not only the web app but also a visual studio code extension!

## What we learned
This is the first time we've tried building a VSCode extension so we learned a lot about how to first serve our models using FastAPI and then to link the ML backend to the extension using axios.
