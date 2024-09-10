# Summarization Methods Documentation

## 1. Basic Summarization from PDF

### Description
- **Extract Text from PDF:** Reads and compiles text from each page of the PDF file.
- **Create Chat Messages:** Constructs a message with the extracted text and a prompt asking for a concise summary.
- **Generate Summary:** Sends the message to the LLM model to generate a basic summary of the speech.
- **Print Summary:** Outputs the generated summary.

### Use Case
- **When to Use:** Useful for straightforward summarization needs where a simple, concise summary is sufficient.
- **Why:** This method is quick and easy, making it ideal for initial summarization when you need a general overview without complex requirements.

## 2. Prompt Template Summarization (Translated to Hindi) from PDF

### Description
- **Extract Text from PDF:** Reads the text content from each page of the PDF file.
- **Create Prompt Template:** Uses a pre-defined template to format the prompt that instructs the LLM model to summarize and translate the content.
- **Format Prompt:** Substitutes the actual text and target language into the template.
- **Generate Summary with Translation:** Sends the formatted prompt to the LLM model to generate a summary and translate it into the specified language.
- **Print Summary:** Outputs the translated summary.

### Use Case
- **When to Use:** Ideal when you need a summary in a specific language, especially for non-English content or for multilingual applications.
- **Why:** This method combines summarization with translation, making it suitable for content that needs to be understood by speakers of different languages.

## 3. StuffDocumentChain Summarization from PDF

### Description
- **Extract Text from PDF:** Reads and compiles text from each page of the PDF.
- **Create Document:** Packages the extracted text into a `Document` object for processing.
- **Use StuffDocumentChain:** Applies the StuffDocumentChain summarization method to generate a summary of the entire document in a single pass.
- **Print Summary:** Outputs the generated summary.

### Use Case
- **When to Use:** Effective for summarizing entire documents where a single, cohesive summary is needed.
- **Why:** This method is straightforward and useful when you want to quickly generate a summary of a complete document without the need for complex processing.

## 4. Map-Reduce Summarization from PDF

### Description
- **Extract Text from PDF:** Collects and compiles text from all pages of the PDF.
- **Split Text into Chunks:** Divides the extracted text into manageable chunks.
- **Use Map-Reduce Summarization:** Summarizes each chunk individually and then combines these summaries to create a final summary of the entire document.
- **Print Summary:** Outputs the final combined summary.

### Use Case
- **When to Use:** Ideal for large documents where processing the entire text in one go is impractical due to size.
- **Why:** This method breaks down the document into smaller parts, making it easier to handle large amounts of text and produce a summary that reflects the entire content.

## 5. Map-Reduce Summarization with Custom Prompts from PDF

### Description
- **Extract Text from PDF:** Reads and gathers text from each page of the PDF.
- **Split Text into Chunks:** Divides the text into smaller chunks.
- **Create Custom Prompts:** Uses specific prompts for summarizing each chunk and for combining the summaries, allowing for tailored summarization.
- **Generate Summary:** Applies the Map-Reduce summarization method with the custom prompts to generate the final summary.
- **Print Summary:** Outputs the summary created with custom prompts.

### Use Case
- **When to Use:** Useful when you need more control over the summarization process, such as incorporating specific instructions or focusing on key aspects.
- **Why:** Custom prompts allow for tailored summarization that can better fit particular needs or stylistic requirements, offering greater flexibility compared to standard methods.

## 6. RefineChain Summarization from PDF

### Description
- **Extract Text from PDF:** Extracts and compiles text from the entire PDF.
- **Split Text into Chunks:** Divides the text into chunks suitable for processing.
- **Use RefineChain Summarization:** Applies the RefineChain method to iteratively improve and refine the summary from the chunks.
- **Print Summary:** Outputs the refined final summary.

### Use Case
- **When to Use:** Best for documents requiring a high-quality, polished summary that captures nuanced details and provides a refined output.
- **Why:** The iterative refinement process helps in producing a more coherent and polished summary, making it suitable for high-stakes content where clarity and detail are crucial.
