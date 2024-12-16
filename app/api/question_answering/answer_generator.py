from typing import List, Dict
import openai
from app.service.log_client import logger
from .context_optimizer import ContextOptimizer

async def process_single_batch(
    question: str, 
    contexts: List[dict], 
    model: str, 
    conversation_summary: str = None, 
    previous_findings: str = None,
    optimizer: ContextOptimizer = None
) -> dict:
    try:
        if optimizer:
            # Calculate available tokens for context
            system_prompt_tokens = optimizer.count_tokens("\n".join([
                "You are a helpful AI assistant answering questions based on the provided context.",
                conversation_summary or "",
                previous_findings or "",
                "Rules and formatting instructions..."
            ]))
            question_tokens = optimizer.count_tokens(question)
            reserved_tokens = 500  # For completion
            available_tokens = optimizer.max_tokens - system_prompt_tokens - question_tokens - reserved_tokens
            contexts = optimizer.optimize_batch(contexts, available_tokens)
        
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(
                f"""[CONTENT_{i}]
SOURCE_TYPE: {ctx['source_type'].upper()}
SOURCE: {ctx['filename']}
TEXT: {ctx['text']}
END_CONTENT_{i}"""
            )
        
        context_text = "\n\n".join(formatted_contexts)
        
        # First determine if the answer can be found in a single source
        specificity_check = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Analyze if the question can be answered completely from a single source or requires multiple sources."},
                {"role": "user", "content": f"Question: {question}\n\nContexts:\n{context_text}"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        single_source = "single source" in specificity_check.choices[0].message.content.lower()
        
        prompt_parts = ["You are a helpful AI assistant answering questions based on the provided context."]
        
        if conversation_summary:
            prompt_parts.append(f"Previous conversation context:\n{conversation_summary}")
        
        if previous_findings:
            prompt_parts.append(f"Previous findings:\n{previous_findings}")
        
        system_prompt = "\n\n".join(prompt_parts) + f"""
Rules:
1. Base answers ONLY on the provided content blocks
2. If the answer isn't in the context, say so
3. {'Use only one most relevant source for this specific question' if single_source else 'Use all relevant sources to provide a complete answer'}
4. Include specific content references
5. Always cite the source documents used in your answer

After your answer, list the sources used in this format:
<SOURCES_USED>
CONTENT_1: [CLIENT] filename.pdf
CONTENT_2: [GLOBAL] example.com
</SOURCES_USED>"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"}
        ]

        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        
        # Extract sources used
        sources = []
        if "<SOURCES_USED>" in answer:
            answer_parts = answer.split("<SOURCES_USED>")
            main_answer = answer_parts[0].strip()
            sources_text = answer_parts[1].split("</SOURCES_USED>")[0].strip()
            
            content_refs = set()
            for line in sources_text.splitlines():
                if ":" in line:
                    content_num = line.split(":")[0].strip()
                    if content_num.startswith("CONTENT_"):
                        try:
                            idx = int(content_num.replace("CONTENT_", "")) - 1
                            if idx < len(contexts):
                                sources.append(contexts[idx])
                        except ValueError:
                            continue
        else:
            main_answer = answer
            
        if not sources:
            # Fallback: check content references in the answer
            for i, ctx in enumerate(contexts, 1):
                if f"CONTENT_{i}" in main_answer:
                    sources.append(contexts[i-1])

        return {
            "answer": main_answer,
            "sources": sources[:1] if single_source else sources,
            "needs_clarification": len(sources) > 1 and not single_source
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise

async def process_contexts_in_batches(
    question: str, 
    contexts: List[dict], 
    model: str, 
    conversation_context: List[dict], 
    batch_size: int = 4
) -> Dict:
    try:
        # Check if this is a followup question
        if conversation_context:
            followup_check = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": "Determine if this question is a followup to the previous conversation and requires previous context to answer accurately."},
                    {"role": "user", "content": f"Previous conversation:\n{str(conversation_context)}\n\nNew question: {question}"}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            is_followup = "yes" in followup_check.choices[0].message.content.lower()
            
            if is_followup:
                # Get sources from previous conversation
                prev_sources = []
                for msg in conversation_context:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        if "sources" in msg:
                            prev_sources.extend(msg["sources"])
                
                # Combine previous sources with current contexts
                if prev_sources:
                    # Create a set of unique filenames from previous sources
                    prev_filenames = {src["filename"] for src in prev_sources}
                    
                    # Filter current contexts to include previous sources and new relevant contexts
                    contexts = [
                        ctx for ctx in contexts 
                        if ctx["filename"] in prev_filenames or 
                        any(text.lower() in ctx["text"].lower() 
                            for text in [question.lower()])
                    ]
        else:
            is_followup = False

        optimizer = ContextOptimizer(model)
        
        if not contexts:
            return {
                "answer": "I cannot find relevant information to answer your question.",
                "sources": [],
                "needs_clarification": False,
                "is_followup": is_followup
            }

        all_answers = []
        all_sources = []
        needs_clarification = False

        # Process conversation history
        history_tokens = optimizer.count_tokens(str(conversation_context))
        if history_tokens > optimizer.max_tokens // 3:
            initial_prompt = {
                "role": "system",
                "content": "Summarize the key points from this conversation history, focusing on information relevant to the current question."
            }
            messages = [initial_prompt, *conversation_context[-2:]]
        else:
            initial_prompt = {
                "role": "system",
                "content": "Review the conversation history and note key points relevant to the current question."
            }
            messages = [initial_prompt, *conversation_context]
            
        messages.append({"role": "user", "content": f"Current question: {question}"})
        
        history_response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        context_summary = history_response.choices[0].message.content

        # Prioritize contexts
        client_contexts = [ctx for ctx in contexts if ctx["source_type"] == "client"]
        global_contexts = [ctx for ctx in contexts if ctx["source_type"] == "global"]
        
        # Process client contexts first
        for i in range(0, len(client_contexts), batch_size):
            batch = client_contexts[i:i + batch_size]
            batch_response = await process_single_batch(
                question=question,
                contexts=batch,
                model=model,
                conversation_summary=context_summary if i == 0 else None,
                previous_findings="\n".join(all_answers) if all_answers else None,
                optimizer=optimizer
            )
            all_answers.append(batch_response["answer"])
            all_sources.extend(batch_response["sources"])
            needs_clarification = needs_clarification or batch_response["needs_clarification"]

        # Only process global contexts if needed
        if not all_answers or "I cannot find relevant information" in all_answers[-1]:
            for i in range(0, len(global_contexts), batch_size):
                batch = global_contexts[i:i + batch_size]
                batch_response = await process_single_batch(
                    question=question,
                    contexts=batch,
                    model=model,
                    previous_findings="\n".join(all_answers) if all_answers else None,
                    optimizer=optimizer
                )
                all_answers.append(batch_response["answer"])
                all_sources.extend(batch_response["sources"])

        # Create final summary
        final_prompt = f"""Synthesize a coherent answer from these findings:
{'-' * 40}
{chr(10).join(all_answers)}
{'-' * 40}

Create a clear, non-repetitive response that addresses the question: {question}
Include all relevant source references in your answer.
"""
        
        if optimizer.count_tokens(final_prompt) > optimizer.max_tokens // 2:
            all_answers = all_answers[:3]
            final_prompt = f"""Synthesize key points from these findings to answer the question:
{chr(10).join(all_answers)}

Question: {question}
Include all relevant source references in your answer."""

        final_response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Create a coherent summary from multiple findings. Always cite sources used."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        # Ensure we have sources, even for followup questions
        if not all_sources and is_followup:
            # Get sources from conversation context that are relevant to the answer
            answer_text = final_response.choices[0].message.content.lower()
            relevant_sources = [
                src for msg in conversation_context 
                if isinstance(msg, dict) and msg.get("role") == "assistant"
                for src in msg.get("sources", [])
                if any(term.lower() in answer_text for term in [
                    src.get("filename", ""),
                    src.get("text", "")[:100]
                ])
            ]
            all_sources.extend(relevant_sources)

        # Deduplicate sources while preserving order
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = (s["filename"], s.get("document_id", ""))
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "answer": final_response.choices[0].message.content,
            "sources": unique_sources,
            "needs_clarification": needs_clarification,
            "is_followup": is_followup
        }

    except Exception as e:
        logger.error(f"Error processing context batches: {str(e)}")
        raise

async def generate_question_suggestions(
    contexts: List[dict], 
    n_suggestions: int, 
    model: str,
    original_question: str,
    needs_clarification: bool
) -> List[str]:
    try:
        if not contexts:
            return []
            
        formatted_contexts = [
            f"""Content: {chunk["text"]}
Source: {chunk["filename"]} ({chunk["source_type"]})
---""" for chunk in contexts[:5]
        ]
        
        context = "\n".join(formatted_contexts)
        
        prompt_addition = """
6. If multiple client sources were found, suggest more specific questions to help narrow down the information source.""" if needs_clarification else ""

        prompt = f"""Based on ONLY the provided content, generate {n_suggestions} questions.

Content:
{context}

Rules:
1. Questions must be answerable using ONLY the provided content
2. Questions should be different from: "{original_question}"
3. Format as numbered list
4. Questions should be relevant and meaningful
5. Questions should explore different aspects of the content{prompt_addition}"""
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": "Generate focused follow-up questions based on the content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        
        questions = [
            line.split('.', 1)[1].strip()
            for line in response.choices[0].message.content.strip().split('\n')
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, n_suggestions + 1))
        ]
        
        return questions[:n_suggestions]
        
    except Exception as e:
        logger.error(f"Question suggestion generation failed: {str(e)}")
        raise