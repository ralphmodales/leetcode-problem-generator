import os
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LeetCodeProblemGenerator:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.difficulty_levels = ["easy", "medium", "hard"]
        self.problem_types = [
            "arrays",
            "strings",
            "linked-lists",
            "trees",
            "graphs",
            "dynamic-programming",
            "sorting",
            "searching",
            "hash-tables",
            "recursion",
            "backtracking",
            "greedy",
            "binary-search",
            "stacks",
            "queues",
            "heap",
            "math",
            "bit-manipulation",
        ]
        self.languages = ["python", "java", "javascript", "cpp", "go"]
        self.site_url = os.getenv("SITE_URL", "localhost:3000")
        self.site_name = os.getenv("SITE_NAME", "LeetCode Generator")
        self.output_dir = Path("generated_problems")
        self.output_dir.mkdir(exist_ok=True)

    def generate_problem(
        self,
        difficulty: Optional[str] = None,
        problem_type: Optional[str] = None,
        constraints: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> Dict:
        if not difficulty:
            difficulty = random.choice(self.difficulty_levels)
        if not problem_type:
            problem_type = random.choice(self.problem_types)
        if not languages:
            languages = ["python", "java", "javascript"]

        prompt = f"""Generate a complete LeetCode-style problem that is NOT on LeetCode with the following attributes:
- Difficulty: {difficulty}
- Topic: {problem_type}
{f"- Additional constraints: {constraints}" if constraints else ""}

You MUST include ALL of the following sections in your response:

1. A clear, concise title for the problem
2. A detailed problem description with all necessary context
3. At least 3 comprehensive examples with inputs, outputs, and clear explanations
4. Specific constraints on input size, value ranges, and edge cases
5. Starter code templates in {', '.join(languages)}
6. Time and space complexity expectations

IMPORTANT: Format your response as a JSON object with the following structure exactly:

{{
  "title": "Your Problem Title",
  "difficulty": "{difficulty}",
  "problem_type": "{problem_type}",
  "description": "Detailed problem description here...",
  "examples": [
    {{
      "input": "Specific input format",
      "output": "Expected output",
      "explanation": "Detailed explanation of this example"
    }},
    {{
      "input": "Another input example",
      "output": "Another output example",
      "explanation": "Explanation for this example"
    }},
    {{
      "input": "Third input example",
      "output": "Third output example",
      "explanation": "Explanation for this example"
    }}
  ],
  "constraints": [
    "1 <= array length <= 10^5",
    "-10^9 <= nums[i] <= 10^9",
    "Other specific constraints..."
  ],
  "starter_code": {{
    "python": "def solution(params):\\n    # Your code here\\n    pass",
    "java": "class Solution {{\\n    public ReturnType solution(ParamType params) {{\\n        // Your code here\\n    }}\\n}}",
    "javascript": "function solution(params) {{\\n    // Your code here\\n}}",
    "other_languages": "..."
  }},
  "expected_time_complexity": "O(X) where X is...",
  "expected_space_complexity": "O(Y) where Y is..."
}}
"""

        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model="meta-llama/llama-4-scout:free",
                messages=[{"role": "user", "content": prompt}],
            )

            response_content = completion.choices[0].message.content
            print(f"Raw response preview: {response_content[:100]}...")

            try:
                problem = json.loads(response_content)
                self._save_problem(problem)
                return problem
            except json.JSONDecodeError:
                cleaned_content = self._clean_json_response(response_content)
                try:
                    problem = json.loads(cleaned_content)
                    self._save_problem(problem)
                    return problem
                except json.JSONDecodeError:
                    problem = self._extract_problem_manually(
                        response_content, difficulty, problem_type, languages
                    )
                    self._save_problem(problem)
                    return problem

        except Exception as e:
            print(f"Error generating problem: {str(e)}")
            return {"error": str(e)}

    def _clean_json_response(self, content: str) -> str:
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()

        json_pattern = r"({[\s\S]*})"
        match = re.search(json_pattern, content)
        if match:
            json_candidate = match.group(1)
            open_braces = json_candidate.count("{")
            close_braces = json_candidate.count("}")
            if open_braces > close_braces:
                json_candidate += "}" * (open_braces - close_braces)
            return json_candidate

        return content

    def _extract_problem_manually(
        self, content: str, difficulty: str, problem_type: str, languages: List[str]
    ) -> Dict:
        problem = {"difficulty": difficulty, "problem_type": problem_type}

        title_patterns = [
            r'title["\s:]+([^"]+?)["\s,}]',
            r"Title:?\s*(.+?)[\n\r]",
            r"#\s*(.+?)[\n\r]",
            r"Problem:?\s*(.+?)[\n\r]",
        ]

        for pattern in title_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                problem["title"] = match.group(1).strip()
                break

        if "title" not in problem:
            problem["title"] = (
                f"{difficulty.capitalize()} {problem_type.capitalize()} Problem"
            )

        desc_patterns = [
            r'description["\s:]+([^"]+?)["\s,}]',
            r"Description:?\s*(.+?)(?=\n\s*\n|Examples|Example 1|Input|Output)",
            r"Problem Statement:?\s*(.+?)(?=\n\s*\n|Examples|Example 1|Input|Output)",
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                problem["description"] = match.group(1).strip()
                break

        if "description" not in problem:
            match = re.search(
                r"^(.*?)(?=\n\s*\n|Examples|Example 1|Input|Output)", content, re.DOTALL
            )
            if match:
                problem["description"] = match.group(1).strip()

        examples = []

        example_blocks = re.findall(
            r"Example\s*\d+:?\s*(.+?)(?=\n\s*\n|Example\s*\d+:|Constraints|Solution|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )

        if example_blocks:
            for block in example_blocks:
                example = {}

                input_match = re.search(
                    r"Input:?\s*(.+?)(?=\n\s*Output|$)",
                    block,
                    re.DOTALL | re.IGNORECASE,
                )
                if input_match:
                    example["input"] = input_match.group(1).strip()

                output_match = re.search(
                    r"Output:?\s*(.+?)(?=\n\s*Explanation|$)",
                    block,
                    re.DOTALL | re.IGNORECASE,
                )
                if output_match:
                    example["output"] = output_match.group(1).strip()

                explanation_match = re.search(
                    r"Explanation:?\s*(.+?)$", block, re.DOTALL | re.IGNORECASE
                )
                if explanation_match:
                    example["explanation"] = explanation_match.group(1).strip()

                if example:
                    examples.append(example)

        if not examples and example_blocks:
            for i, block in enumerate(example_blocks):
                examples.append(
                    {
                        "input": f"See example {i+1}",
                        "output": f"See example {i+1}",
                        "explanation": block.strip(),
                    }
                )

        if not examples:
            examples = [
                {
                    "input": "Example input would be shown here",
                    "output": "Example output would be shown here",
                    "explanation": "Examples couldn't be properly extracted",
                }
            ]

        problem["examples"] = examples

        constraints_pattern = r"Constraints:?\s*(.+?)(?=\n\s*\n|Solution|Approach|$)"
        match = re.search(constraints_pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            constraints_text = match.group(1).strip()
            constraints = []
            for line in constraints_text.split("\n"):
                clean_line = line.strip()
                if (
                    clean_line
                    and not clean_line.startswith("#")
                    and not clean_line.lower().startswith("constraints")
                ):
                    clean_line = re.sub(r"^[-•*]\s*", "", clean_line)
                    constraints.append(clean_line)

            problem["constraints"] = constraints
        else:
            if difficulty == "easy":
                problem["constraints"] = [
                    "1 <= input size <= 10^3",
                    "-10^3 <= values <= 10^3",
                ]
            elif difficulty == "medium":
                problem["constraints"] = [
                    "1 <= input size <= 10^4",
                    "-10^4 <= values <= 10^4",
                ]
            else:  # hard
                problem["constraints"] = [
                    "1 <= input size <= 10^5",
                    "-10^9 <= values <= 10^9",
                ]

        starter_code = {}
        for lang in languages:
            lang_pattern = (
                rf"{lang}\s*[:\(]?.*?```(?:python|java|javascript|cpp|go)?\s*(.*?)```"
            )
            match = re.search(lang_pattern, content, re.DOTALL | re.IGNORECASE)

            if match:
                starter_code[lang] = match.group(1).strip()
            else:
                code_block_pattern = r"```(?:python|java|javascript|cpp|go)?\s*(.*?)```"
                match = re.search(code_block_pattern, content, re.DOTALL)

                if match:
                    code = match.group(1).strip()
                    if lang == "python":
                        starter_code[lang] = (
                            "def solution(params):\n    # Your code here\n    pass"
                        )
                    elif lang == "java":
                        starter_code[lang] = (
                            "class Solution {\n    public ReturnType solution(ParamType params) {\n        // Your code here\n        return null;\n    }\n}"
                        )
                    elif lang == "javascript":
                        starter_code[lang] = (
                            "function solution(params) {\n    // Your code here\n    return null;\n}"
                        )
                    elif lang == "cpp":
                        starter_code[lang] = (
                            "class Solution {\npublic:\n    ReturnType solution(ParamType params) {\n        // Your code here\n        return {};\n    }\n};"
                        )
                    elif lang == "go":
                        starter_code[lang] = (
                            "func solution(params ParamType) ReturnType {\n    // Your code here\n    return nil\n}"
                        )
                else:
                    if lang == "python":
                        starter_code[lang] = (
                            "def solution(params):\n    # Your code here\n    pass"
                        )
                    elif lang == "java":
                        starter_code[lang] = (
                            "class Solution {\n    public ReturnType solution(ParamType params) {\n        // Your code here\n        return null;\n    }\n}"
                        )
                    elif lang == "javascript":
                        starter_code[lang] = (
                            "function solution(params) {\n    // Your code here\n    return null;\n}"
                        )
                    elif lang == "cpp":
                        starter_code[lang] = (
                            "class Solution {\npublic:\n    ReturnType solution(ParamType params) {\n        // Your code here\n        return {};\n    }\n};"
                        )
                    elif lang == "go":
                        starter_code[lang] = (
                            "func solution(params ParamType) ReturnType {\n    // Your code here\n    return nil\n}"
                        )

        problem["starter_code"] = starter_code

        time_pattern = r"Time\s*Complexity:?\s*(.+?)(?=\n\s*\n|Space|$)"
        match = re.search(time_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            problem["expected_time_complexity"] = match.group(1).strip()
        else:
            problem["expected_time_complexity"] = "O(n) - Estimated"

        space_pattern = r"Space\s*Complexity:?\s*(.+?)(?=\n\s*\n|$)"
        match = re.search(space_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            problem["expected_space_complexity"] = match.group(1).strip()
        else:
            problem["expected_space_complexity"] = "O(n) - Estimated"

        return problem

    def _save_problem(self, problem: Dict) -> None:
        title = problem.get("title", "untitled_problem")
        title = re.sub(r"[^\w\s-]", "", title).strip().lower()
        title = re.sub(r"[-\s]+", "_", title)

        difficulty = problem.get("difficulty", "unknown")
        filename = f"{difficulty}_{title}.json"

        with open(self.output_dir / filename, "w") as f:
            json.dump(problem, f, indent=2)
        print(f"Problem saved to {self.output_dir / filename}")

    def batch_generate(
        self,
        count: int = 5,
        difficulty: Optional[str] = None,
        problem_type: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> List[Dict]:
        problems = []
        for i in range(count):
            print(f"\nGenerating problem {i+1}/{count}...")
            problem = self.generate_problem(difficulty, problem_type, None, languages)
            problems.append(problem)
        return problems


def main():
    generator = LeetCodeProblemGenerator()

    import argparse

    parser = argparse.ArgumentParser(description="Generate LeetCode style problems")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of problems to generate"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], help="Problem difficulty"
    )
    parser.add_argument("--type", dest="problem_type", help="Problem type/category")
    parser.add_argument(
        "--languages",
        help="Comma-separated list of languages for starter code",
        default="python,java,javascript",
    )
    parser.add_argument("--constraints", help="Additional constraints for the problem")
    parser.add_argument(
        "--output", help="Custom output directory", default="generated_problems"
    )

    args = parser.parse_args()

    if args.output:
        generator.output_dir = Path(args.output)
        generator.output_dir.mkdir(exist_ok=True)

    languages = [lang.strip().lower() for lang in args.languages.split(",")]

    if args.count > 1:
        problems = generator.batch_generate(
            args.count, args.difficulty, args.problem_type, languages
        )
        print(f"\nGenerated {len(problems)} problems in {generator.output_dir}")
    else:
        problem = generator.generate_problem(
            args.difficulty, args.problem_type, args.constraints, languages
        )
        if "title" in problem:
            print(f"\nGenerated problem: {problem['title']}")
            print(f"Difficulty: {problem['difficulty']}")
            print(f"Problem type: {problem['problem_type']}")
            print(
                f"Has starter code: {', '.join(problem.get('starter_code', {}).keys())}"
            )
            print(f"Has examples: {len(problem.get('examples', []))}")
            print(f"Has constraints: {len(problem.get('constraints', []))}")
            print(f"Saved to: {generator.output_dir}")
        else:
            print(f"\nGenerated problem with raw content (parsing failed)")
            print(f"Saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()
