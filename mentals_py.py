import os
import re
import json
import toml
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {e}")
            raise

class GenFile:
    @staticmethod
    def remove_comment_lines(text: str) -> str:
        return "\n".join(line for line in text.split("\n") if not line.strip().startswith("///"))

    @staticmethod
    def parse_directive_input(text: str, default_input: str) -> Tuple[str, str]:
        lines = text.split("\n")
        input_prompt = default_input
        new_text = []
        for line in lines:
            if line.strip().lower().startswith("## input:"):
                input_prompt = line.split(":", 1)[1].strip()
            else:
                new_text.append(line)
        return "\n".join(new_text), input_prompt

    @staticmethod
    def parse_directive_use(text: str) -> Tuple[str, List[str]]:
        lines = text.split("\n")
        use_names = []
        new_text = []
        in_use_section = False
        use_section = ""
        for line in lines:
            if in_use_section:
                if line.strip() and not line.startswith(" "):
                    use_names.extend([name.strip() for name in use_section.split(",")])
                    in_use_section = False
                    new_text.append(line)
                else:
                    use_section += " " + line.strip()
            elif line.strip().lower().startswith("## use:"):
                in_use_section = True
                use_section = line.split(":", 1)[1].strip()
            else:
                new_text.append(line)
        if use_section:
            use_names.extend([name.strip() for name in use_section.split(",")])
        return "\n".join(new_text), use_names

    @staticmethod
    def parse_directive_keep_context(text: str) -> Tuple[str, bool]:
        lines = text.split("\n")
        keep_context = True
        new_text = []
        for line in lines:
            if line.strip().lower().startswith("## keep_context:"):
                keep_context = line.split(":", 1)[1].strip().lower() == "true"
            else:
                new_text.append(line)
        return "\n".join(new_text), keep_context

    @staticmethod
    def parse_directive_max_context(text: str) -> Tuple[str, int]:
        lines = text.split("\n")
        max_context = 0
        new_text = []
        for line in lines:
            if line.strip().lower().startswith("## max_context:"):
                max_context = int(line.split(":", 1)[1].strip())
            else:
                new_text.append(line)
        return "\n".join(new_text), max_context

    @staticmethod
    def parse_variable_sections(gen_content: str) -> Tuple[str, Dict[str, str]]:
        variables = {}
        var_section_pattern = r'\{\{(\w+)\}\}([\s\S]*?)\{\{/\1\}\}'
        for match in re.finditer(var_section_pattern, gen_content):
            key, value = match.groups()
            value = value.strip()
            variables[key] = value
            gen_content = gen_content.replace(match.group(0), "")
        return gen_content.strip(), variables

    @staticmethod
    def parse_instructions(gen_content: str) -> Dict[str, Dict]:
        instructions = {}
        section_pattern = r'# (\w+)\s*\n([\s\S]*?)(?=\n# \w+\s*\n|$)'
        for match in re.finditer(section_pattern, gen_content):
            label, prompt = match.groups()
            prompt = prompt.strip()
            default_input = "Content in a plain text to send to the function."
            prompt, input_prompt = GenFile.parse_directive_input(prompt, default_input)
            prompt, use = GenFile.parse_directive_use(prompt)
            prompt, keep_context = GenFile.parse_directive_keep_context(prompt)
            prompt, max_context = GenFile.parse_directive_max_context(prompt)
            instructions[label] = {
                "label": label,
                "prompt": prompt,
                "input_prompt": input_prompt,
                "temperature": 0.1,
                "use": use,
                "keep_context": keep_context,
                "max_context": max_context
            }
        return instructions

    @staticmethod
    def load_from_file(filename: str) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        if not filename.endswith(".gen"):
            raise ValueError("Only .gen files are supported")
        
        try:
            with open(filename, "r") as file:
                gen_file = file.read()
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except IOError as e:
            logging.error(f"Error reading file {filename}: {e}")
            raise
        
        gen_file = GenFile.remove_comment_lines(gen_file)
        gen_file, variables = GenFile.parse_variable_sections(gen_file)
        instructions = GenFile.parse_instructions(gen_file)
        
        return variables, instructions

class AgentExecutor:
    def __init__(self, config: Dict, api_key: str):
        self.config = config
        self.short_term_memory: Dict[str, str] = {}
        self.agent_executor_state: Dict[str, str] = {}
        self.native_instructions: Dict[str, Dict] = {}
        self.agent_instructions: Dict[str, Dict] = {}
        self.working_memory: List[Dict[str, str]] = []
        self.working_contexts: Dict[str, List[Dict[str, str]]] = {}
        self.instructions_call_stack: List[str] = []
        self.instructions: Dict[str, Dict] = {}
        self.openai_client = OpenAIClient(api_key)

    def set_state_variable(self, name: str, value: str):
        self.agent_executor_state[name] = value

    def init_native_tools(self, file_path: str) -> bool:
        try:
            with open(file_path, "r") as file:
                self.native_instructions = toml.load(file)["instruction"]
            return True
        except FileNotFoundError:
            logging.error(f"Native tools file not found: {file_path}")
        except toml.TomlDecodeError as e:
            logging.error(f"Error parsing native tools file: {e}")
        except KeyError:
            logging.error("Missing 'instruction' key in native tools file")
        return False

    def init_agent(self, inst: Dict[str, Dict]):
        self.instructions = inst

    def run_agent_thread(self, entry_instruction: str, input_text: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        if context is None:
            context = []
        
        self.working_memory = context
        instruction = self.instructions[entry_instruction]
        
        # Prepare messages for OpenAI API
        messages = [{"role": "system", "content": instruction["prompt"]}]
        messages.extend(self.working_memory)
        messages.append({"role": "user", "content": input_text})
        
        # Generate response using OpenAI API
        content = self.openai_client.generate_response(messages, temperature=instruction["temperature"])
        
        # Update working memory if needed
        if instruction["keep_context"]:
            self.working_memory.append({"role": "assistant", "content": content})
            if instruction["max_context"] > 0 and len(self.working_memory) > instruction["max_context"]:
                self.working_memory = self.working_memory[-instruction["max_context"]:]
        
        # Process 'use' instructions if any
        for use_instruction in instruction["use"]:
            if use_instruction in self.instructions:
                content = self.run_agent_thread(use_instruction, content, self.working_memory)
        
        return content

def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_platform_info() -> str:
    import platform
    return f"OS: {platform.system()} {platform.release()}"

def main():
    try:
        # Check if an input argument is provided
        if len(sys.argv) < 2:
            print("Usage: python3 mentals_py.py '<input_text>'")
            sys.exit(1)

        input_text = sys.argv[1]

        # Load config
        try:
            with open("config.toml", "r") as config_file:
                config = toml.load(config_file)
        except FileNotFoundError:
            logging.error("Config file not found: config.toml")
            raise
        except toml.TomlDecodeError as e:
            logging.error(f"Error parsing config file: {e}")
            raise

        # Initialize AgentExecutor with the API key from config
        api_key = config.get("llm", {}).get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in config file")
        
        agent_executor = AgentExecutor(config, api_key)

        # Set state variables
        agent_executor.set_state_variable("current_date", get_current_date())
        agent_executor.set_state_variable("platform_info", get_platform_info())

        # Initialize native tools
        if not agent_executor.init_native_tools("native_tools.toml"):
            raise RuntimeError("Failed to init native tools")

        # Load agent file
        gen_file = GenFile()
        filename = "agents/memory.gen"
        variables, instructions = gen_file.load_from_file(filename)

        logging.info("Loaded variables:")
        logging.info(json.dumps(variables, indent=2))
        logging.info("Loaded instructions:")
        logging.info(json.dumps(instructions, indent=2))

        # Initialize agent
        agent_executor.init_agent(instructions)

        logging.info(f"Processing input: {input_text}")
        result = agent_executor.run_agent_thread("root", input_text)
        logging.info(f"Agent response: {result}")
        
        print(result)  # Print the final result to stdout

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()