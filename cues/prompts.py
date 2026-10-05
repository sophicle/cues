"""The prompt bank. Every input is plain text; a chat template is applied only for chat models
(--chat), because a template is itself an opening cue.

Ported from sophicle/reason registers/prompts.py; the strings are byte-identical.

  boxed     the Qwen2.5-Math instruction sentence in the Hendrycks MATH Problem:/Solution: layout
  rlzero    the prompt Ai2 trained Olmo-3-7B-RL-Zero-Math with (Olmo 3 report, Figure 37); answers
            go on an "Answer:" line, so this prompt is graded by that line (see cues.grading)
  minerva4  the OLMES/Minerva 4-shot prompt used for Olmo 3's base-model math evals; our stop
            strings allow multi-paragraph solutions where OLMES stops at the first blank line
  chat      the user turn for Instruct/Think models: the question plus the boxed instruction; the
            model's own chat template supplies the roles

STOPS halt a base model that finishes its solution and starts writing a fresh problem; without
them the grader reads whatever the model drifted into. GRADING names the answer contract each
prompt asks for.
"""

BOXED = ("Problem: {q}\nPlease reason step by step, and put your final answer "
         "within \\boxed{{}}.\nSolution:")

RLZERO = ("Solve the following math problem step by step.\n"
          "The last line of your response should be the answer to the problem "
          "in form Answer: $Answer (without quotes) where $Answer is the answer "
          "to the problem.\n{q}\n"
          "Remember to put your answer on its own line after \"Answer:\"")

CHAT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

_MINERVA_SHOTS = [
    ("Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
     "The expressions inside each square root must be non-negative. Therefore, "
     "$x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the "
     "denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. "
     "Therefore, the domain of the expression is $\\boxed{[2,5)}$.\n"
     "Final Answer: The final answer is $[2,5)$. I hope it is correct."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
     "$\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})"
     "(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\n"
     "Final Answer: The final answer is $24$. I hope it is correct."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two "
     "15-pound weights instead, how many times must Terrell lift them in order "
     "to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
     "$2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound "
     "weights instead for $n$ times, he will lift a total of "
     "$2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, "
     "we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n"
     "\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\n"
     "Final Answer: The final answer is $16$. I hope it is correct."),
    ("If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n"
     "6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are "
     "both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n"
     "$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n"
     "$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\n"
     "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."),
]
MINERVA_PROMPT = "".join(
    "Problem:\n%s\n\nSolution: %s\n\n" % (q.replace("{", "{{").replace("}", "}}"),
                                           sol.replace("{", "{{").replace("}", "}}"))
    for q, sol in _MINERVA_SHOTS) + "Problem:\n{q}\n\nSolution:"

MINERVA4 = MINERVA_PROMPT

PROMPTS = {"boxed": BOXED, "rlzero": RLZERO, "minerva4": MINERVA4, "chat": CHAT}

# rlzero also stops on the trainer's stop string (a base model re-emitting the instruction), so
# training rollouts and evaluation rollouts end the same way.
STOPS = {"boxed": ["\nProblem:", "\nProblem :"],
         "minerva4": ["\nProblem:", "\nProblem :"],
         "rlzero": ["\nProblem:", "\nProblem :", "\nSolve the following math problem"],
         "chat": []}

GRADING = {"boxed": "boxed", "rlzero": "rlzero", "minerva4": "boxed", "chat": "boxed"}


def render(prompt, question, cue=""):
    """The prompt for one problem with the cue appended as raw text. The model's own tokenizer sees
    prompt + cue as one string, in evaluation and in RL alike."""
    return PROMPTS[prompt].format(q=question) + cue
