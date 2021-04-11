import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
import torch
from tree_sitter import Language, Parser
import os
import base64
from io import StringIO

Language.build_library('build/my-languages.so', ['tree-sitter-python', 'tree-sitter-go', 'tree-sitter-javascript', 'tree-sitter-java'])

def build_parser(language):
    LANGUAGE = Language('build/my-languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    return parser

def get_string_from_code(node, lines, code_list):
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
	code_list.append(' '.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]]))
    else:
	code_list.append(lines[line_start][char_start:char_end])

def my_traverse(code, node, code_list):
    lines = code.split('\n')
    if node.child_count == 0:
	get_string_from_code(node, lines, code_list)
    elif node.type == 'string':
	get_string_from_code(node, lines, code_list)
    else:
	for n in node.children:
	    my_traverse(code, n, code_list)
    return ' '.join(code_list)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

@st.cache(allow_output_mutation=True)
def loadmodel(name):
    device = torch.device("cuda")
    model = AutoModelWithLMHead.from_pretrained(name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name, skip_special_tokens=True)
    return (model, tokenizer)

st.set_page_config(page_title="CodeDecoder", page_icon="💻", layout="wide")

def main():
    st.title('Code Decoder')
    language = st.sidebar.selectbox('Choose Language', ('Python', 'Java', 'Javascript', 'Go'))
    format = st.selectbox('Upload or Paste your code over', ('Upload', 'Paste'))
    if language == 'Python':
    	parser = build_parser("python")
    	name = "t5_base_python"
    	python_model, python_tokenizer = loadmodel(name)
    	python_pipeline = SummarizationPipeline(model=python_model, tokenizer=python_tokenizer, device=0)
    	
    	if format == 'Paste':
        	code = st.text_area("Enter code")
        	if code is not None:
        		tree = parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = python_pipeline([tokenized_code])
        		final_code = f"# {out[0]['summary_text']}\n{code}"
        		st.code(final_code, language="python")
        		fname = 'commented_code.py'
        		with open(fname, 'w') as f:
            		f.write(final_code)
          		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)
    	
    	elif format == 'Upload':
        	uploaded_file = st.file_uploader("Upload your python file")
        	if uploaded_file is not None:
        		stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        		code = stringio.read()
        		tree = parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = python_pipeline([tokenized_code])
        		final_code = f"# {out[0]['summary_text']}\n{code}"
        		st.code(final_code, language="python")
        		fname = 'commented_code.py'
        		with open(fname, 'w') as f:
            		f.write(final_code)
        		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)
    
    elif language == 'Go':
    	GO_LANGUAGE = Language('build/my-languages.so', 'go')
    	go_parser = Parser()
    	go_parser.set_language(GO_LANGUAGE)
    	#parser = build_parser("go")
    	name = "t5_base_go"
    	go_model, go_tokenizer = loadmodel(name)
    	go_pipeline = SummarizationPipeline(model=go_model, tokenizer=go_tokenizer, device=0)
    	
    	if format == 'Paste':
        	code = st.text_area("Enter code")
        	if code is not None:
        		tree = go_parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = go_pipeline([tokenized_code])
        		final_code = f"// {out[0]['summary_text']}\n{code}"
        		st.code(final_code, language="go")
          		fname = 'commented_code.go'
        		with open(fname, 'w') as f:
            		f.write(final_code)
        		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)
    	
    	elif format == 'Upload':
        	uploaded_file = st.file_uploader("Upload your go file")
        	if uploaded_file is not None:
        		stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        		code = stringio.read()
        		tree = go_parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = go_pipeline([tokenized_code])
        		st.code(f"// {out[0]['summary_text']}\n{code}", language="go")
        		fname = 'commented_code.go'
        		with open(fname, 'w') as f:
            		f.write(final_code)
        		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)
    
    elif language == 'Javascript':
    	JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
    	js_parser = Parser()
    	js_parser.set_language(JS_LANGUAGE)
    	name = "t5_base_javascript"
    	js_model, js_tokenizer = loadmodel(name)
    	js_pipeline = SummarizationPipeline(model=js_model, tokenizer=js_tokenizer, device=0)
    	
    	if format == 'Paste':
        	code = st.text_area("Enter code")
        	if code:
        	tree = js_parser.parse(bytes(code, "utf8"))
        	code_list=[]
        	tokenized_code = my_traverse(code, tree.root_node, code_list)
        	out = js_pipeline([tokenized_code])
        	final_code = f"// {out[0]['summary_text']}\n{code}"
        	st.code(final_code, language="javascript")
        	fnamejs = 'commented_code.js'
        	with open(fnamejs, 'w') as fjs:
            	fjs.write(final_code)
        	st.markdown(get_binary_file_downloader_html(fnamejs, 'Code'), unsafe_allow_html=True)
    	
    	elif format == 'Upload':
        	uploaded_file = st.file_uploader("Upload your javascript file")
        	if uploaded_file is not None:
        		stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        		code = stringio.read()
        		tree = go_parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = go_pipeline([tokenized_code])
        		st.code(f"// {out[0]['summary_text']}\n{code}", language="javascript")
        		fname = 'commented_code.js'
        		with open(fname, 'w') as f:
            		f.write(final_code)
        		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)
    
    elif language == 'Java':
    	JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
    	java_parser = Parser()
    	java_parser.set_language(JAVA_LANGUAGE)
    	name = "t5_base_java"
    	java_model, java_tokenizer = loadmodel(name)
    	java_pipeline = SummarizationPipeline(model=java_model, tokenizer=java_tokenizer, device=0)
    	
    	if format == 'Paste':
        	code = st.text_area("Enter code")
        	if code:
        		tree = java_parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = java_pipeline([tokenized_code])
        		final_code = f"// {out[0]['summary_text']}\n{code}"
        		st.code(final_code, language="java")
        		fname = 'commented_code.java'
        		with open(fname, 'w') as f:
            		f.write(final_code)
        		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)
    	
    	elif format == 'Upload':
        	uploaded_file = st.file_uploader("Upload your java file")
        	if uploaded_file is not None:
        		stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        		code = stringio.read()
        		tree = go_parser.parse(bytes(code, "utf8"))
        		code_list=[]
        		tokenized_code = my_traverse(code, tree.root_node, code_list)
        		out = go_pipeline([tokenized_code])
        		st.code(f"// {out[0]['summary_text']}\n{code}", language="java")
        		fname = 'commented_code.java'
        		with open(fname, 'w') as f:
            		f.write(final_code)
        		st.markdown(get_binary_file_downloader_html(fname, 'Code'), unsafe_allow_html=True)

if __name__=='__main__':
	main()
