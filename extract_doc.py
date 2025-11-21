from docx import Document

doc = Document('finalprojectpaper.docx')

full_text = []
for para in doc.paragraphs:
    full_text.append(para.text)

text = '\n'.join(full_text)
print(text)

# Save to txt file for easier reading
with open('project_requirements.txt', 'w', encoding='utf-8') as f:
    f.write(text)
