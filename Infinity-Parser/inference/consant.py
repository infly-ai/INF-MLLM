PROMPT = """Please convert the image document into Markdown format, strictly following the requirements below:

1. **Text Processing**
   - Ignore headers and footers, but accurately recognize and extract all other text content from the image document without guessing or inferring missing parts.
   - Convert the recognized text into Markdown format.
   - Preserve the original document structure, including titles, paragraphs, and lists.

2. **Formula Processing**
   - Convert all formulas into LaTeX format.
   - Inline formulas should be enclosed in `$ $`.
     Example: This is an inline formula $E = mc^{2}$.
   - Display (block) formulas should be enclosed in `$$ $$`.
     Example:
     $$\\text{Distance} = \\text{Speed} \\times \\text{Time}$$

3. **Table Processing**
   - Convert all tables into Markdown table format.

4. **Image Processing**
   - Ignore all graphical content in the image document. Do not attempt to describe or convert the images.

5. **Output Format**
   - Ensure the output Markdown document has a clear and organized structure, with appropriate line breaks between elements.
   - For complex layouts, preserve the original structure and formatting as much as possible.

Please strictly adhere to these requirements to ensure accuracy and consistency in the conversion.
"""
