from utils.extract_option import extract_option


cot_template = """You are a professional medical expert to answer the # Question. Please first using the # Retrieved Documents and then answer the question. Your responses will be used for research purposes only, so please have a definite and direct answer.
You should respond in the format <answer>A/B/C/D</answer> (only one option can be chosen) at your response.

# Question
{question}"""

ret_template = """You are a professional medical expert to answer the # Question. Please first using the # Retrieved Documents and then answer the question. Your responses will be used for research purposes only, so please have a definite and direct answer.
You should respond in the format <answer>A/B/C/D</answer> (only one option can be chosen) at your response.

# Retrieved Documents
{documents}

# Question
{question}"""

class Reader():
    def __init__(self, llm, args):
        self.llm = llm
        self.args = args

    def run(self, question, documents):
        if documents is None:
            prompt = cot_template.format(question=question)
        else:
            prompt = ret_template.format(question=question, documents=documents)
        
        format_feedback = None
        for _ in range(5):
            if format_feedback is not None:
                prompt += format_feedback
            format_feedback = None
            llm_output = self.llm.run([prompt], n_seqs=1)[0][0]

            if extract_option(llm_output) not in ["A", "B", "C", "D", "E"]:
                format_feedback = "\n\n# You should think step by step and respond in the format <answer>A/B/C/D</answer> (only one option can be chosen) at the end of your response. Please keep your entire response simple and complete, up to 100 words."
            if format_feedback is None:
                break
        return llm_output
