# SwiftDesk IT Support Assistant - Final Report

## 1. Introduction
This report outlines the design, evaluation, and responsible AI considerations for the SwiftDesk IT Support Assistant. The assistant utilizes the Google Gemini API, LangChain, and a local Chroma DB vector database to draft responses for IT support tickets using Retrieval-Augmented Generation (RAG).

## 2. Prompt Design Choices
During development, we experimented with three distinct prompting strategies to find the optimal balance of tone, clarity, and accuracy:
- **Zero-Shot Prompting:** Served as the baseline. The model was instructed to act as a support agent and reply directly to the issue. It performed reasonably well but sometimes lacked specific, standardized IT protocols.
- **Few-Shot Prompting:** We provided examples of ideal customer issue and agent reply pairs in the prompt. This significantly improved the formatting, tone consistency, and adherence to the company's communication style.
- **Reasoned Prompting (Chain-of-Thought):** The model was asked to first briefly analyze the issue and identify the probable root cause before drafting the response. This improved technical accuracy for complex tickets.
**Decision:** The final system utilizes a dynamic prompt that incorporates the RAG retrieved context as "few-shot-like" examples, combined with clear system instructions on tone and formatting.

## 3. RAG Design
The Retrieval-Augmented Generation (RAG) pipeline was implemented to ground the Gemini model in historical, approved support resolutions:
- **Dataset:** A subset of English tickets from the Kaggle "Customer IT Support - Ticket Dataset" was selected for relevance and quality.
- **Embeddings & Storage:** We generated embeddings for the historical tickets and stored them locally in a Chroma DB vector database for fast access.
- **Retrieval:** When a new ticket arrives, LangChain searches Chroma DB to find the top most similar previous tickets.
- **Generation:** These retrieved tickets are injected into the prompt as "Reference Context". The model is explicitly instructed to base its drafted reply heavily on these successful past resolutions.

## 4. Evaluation Results
The system was evaluated both automatically and manually on a test subset:
- **Automated Evaluation (ROUGE):** We used the `rouge-score` library to compare generated replies against reference human agent replies. The RAG-enabled generation showed improvements in ROUGE-L scores compared to the zero-shot baseline, indicating that retrieving similar historical cases helps the model use the right terminology and steps.
- **Manual Review:** A manual evaluation was conducted on a sample of test cases, checking for four criteria:
  - *Clarity:* High. The generated responses are well-structured and easy for end-users to understand.
  - *Relevance:* High. Thanks to RAG, the troubleshooting steps provided directly address the user's specific problem.
  - *Politeness:* Excellent. The model adhered consistently to the professional tone guidelines.
  - *Safety:* Pass. The model avoided suggesting risky commands or exposing sensitive information.

## 5. Limitations
- **Context Window Constraints:** Extremely long customer email threads or massive log files attached to tickets may need to be truncated, potentially missing key details.
- **Semantic Search False Positives:** Vector search might occasionally retrieve tickets that share similar keywords (e.g., "password reset") but relate to completely different systems, which could confuse the model if not carefully prompted.
- **Local Database Scalability:** While Chroma DB is excellent for this local prototype (~500-1000 tickets), ingesting and querying millions of historical tickets would require migrating to a production-grade cloud vector database.

## 6. Responsible AI Decisions
To ensure the system is safe, ethical, and helpful, several Responsible AI guidelines were implemented, formalized in the `RAI_Config.yaml`:
- **Mandatory Human-in-the-Loop:** All AI generations are explicitly labeled as "DRAFTS" in the Streamlit UI. A human agent must review, edit, and manually send the final message.
- **Anti-Hallucination Measures:** To minimize hallucinations, the prompt strictly instructs the model to state "further investigation is needed" if the RAG context and its general knowledge cannot safely resolve the issue.
- **Escalation Rules:** The system is guided not to provide destructive troubleshooting steps (e.g., deleting a database) and to flag potentially severe incidents (like security breaches) for immediate escalation to a senior engineer.
- **Privacy Protections:** Guidelines state that the model must not generate Personally Identifiable Information (PII) and should abstract away sensitive data.
