import json
from typing import Dict, Any, Tuple
from benchmarks.base_benchmark import BaseBenchmark
from lib.run_query import run_query
from lib.scoring import calculate_score, calculate_score_fullscale, parse_answers, parse_answers_de
from lib.util import safe_dump, remove_revision_instructions, gpu_cleanup
from lib.run_bench_helper_functions import run_test_prompts, format_include_exclude_string

class EQBench(BaseBenchmark):
	def __init__(self, config, args, benchmark_config, runner):
		super().__init__(config, args, benchmark_config, runner)
		self.questions = self.load_questions()
  
	

	def load_questions(self):
		questions_fn = self.get_questions_filename()
		with open(questions_fn, 'r', encoding='utf-8') as f:
			return json.load(f)

	def get_questions_filename(self):
		base_filename = './data/eq_bench_v2_questions_171.json'
		if self.args.v1:
			if self.args.l != "en":
					raise Exception('Error: Only English language is supported for EQ-Bench v1.')
			return './data/eq_bench_v1_questions_60.json'
		
		if self.args.l != 'en':
			base_name, ext = base_filename.rsplit('.', 1)
			return f"{base_name}_{self.args.l}.{ext}"
		
		return base_filename

	def generate_run_index(self):
		components = [
			self.benchmark_config['run_id'],
			"v1" if self.args.v1 else "v2",
			self.args.l,
			self.benchmark_config['model_path'],
			self.benchmark_config['lora_path'],
			self.benchmark_config['prompt_type'],
			self.benchmark_config['quantization'],
			self.benchmark_config['inference_engine'],
			self.benchmark_config['ooba_params'],
			format_include_exclude_string(self.benchmark_config['include_patterns'], self.benchmark_config['exclude_patterns'])
		]
		return "--".join(filter(None, components))

	def run(self):
		for run_iter in range(1, self.benchmark_config['n_iterations'] + 1):
			print(f"Iteration {run_iter} of {self.benchmark_config['n_iterations']}")
			self.initialize_iteration_results(run_iter)
			
			for question_id, question in self.questions.items():
					if self.is_question_completed(question_id, run_iter):
						if self.args.v:
							print(f"Question {question_id} already complete")
						continue
					
					if not self.runner.model and not self.runner.ooba_instance:
						self.runner.initialize_model_or_ooba(self.benchmark_config)
					
					self.process_question(question_id, question, run_iter)

		self.save_results()
  
	def is_question_completed(self, question_id, run_iter):
		return (self.run_index in self.results and
					str(run_iter) in self.results[self.run_index]['iterations'] and
					question_id in self.results[self.run_index]['iterations'][str(run_iter)]['individual_scores'])

	def initialize_iteration_results(self, run_iter):
		if self.run_index not in self.results:
			self.results[self.run_index] = {
					'run_metadata': self.create_run_metadata(),
					'iterations': {}
			}
		self.results[self.run_index]['iterations'][str(run_iter)] = {
			'respondent_answers': {},
			'individual_scores': {},
			'individual_scores_fullscale': {},
			'raw_inference': {}
		}

	def create_run_metadata(self):
		return {
			"run_id": self.benchmark_config['run_id'],
			"benchmark_type": "eq-bench",
			"eq_bench_version": "v1" if self.args.v1 else "v2",
			"language": self.args.l,
			"total_iterations": self.benchmark_config['n_iterations'],
			"inference_engine": self.benchmark_config['inference_engine'],
			"ooba_params": self.benchmark_config['ooba_params'],
			"include_patterns": self.benchmark_config['include_patterns'],
			"exclude_patterns": self.benchmark_config['exclude_patterns'],
			"instruction_template": self.benchmark_config['prompt_type'],
			"model_path": self.benchmark_config['model_path'],
			"lora_path": self.benchmark_config['lora_path'],
			"bitsandbytes_quant": self.benchmark_config['quantization']
		}

	def process_question(self, question_id, question, run_iter):
		prompt = self.prepare_prompt(question['prompt'])
		ref = question['reference_answer']
		ref_fullscale = question.get('reference_answer_fullscale')

		tries = 0
		success = False
		temp = 0.01
		while tries < self.args.r and not success:
			try:
					inference = self.run_inference(prompt, temp)
					
					if self.args.v:
						print('\n' + inference)
						print('________________')

					scores, parsed_answers = self.parse_and_score(inference, ref, ref_fullscale)
					self.store_results(question_id, scores, parsed_answers, inference, run_iter)
					success = True
			except Exception as e:
					print(e)
					tries += 1
					temp += 0.15
					if tries < self.args.r:
						print('Retrying...')

		if not success:
			print(f"Failed to get a valid response for question {question_id} after {self.args.r} attempts.")


	def prepare_prompt(self, prompt):
		if not self.args.v1 and not self.args.revise:
			return remove_revision_instructions(prompt, self.args.l)
		return prompt

	def run_inference(self, prompt, temp):
		return run_query(
			self.benchmark_config['model_path'],
			self.benchmark_config['prompt_type'],
			prompt,
			[],  # history
			600 if self.args.revise else 60,  # max_tokens
			self.runner.model,
			self.runner.tokenizer,
			temp,
			self.benchmark_config['inference_engine'],
			self.runner.ooba_instance,
			self.config.get_bool('Oobabooga config', 'automatically_launch_ooba'),
			self.config.get_int('Oobabooga config', 'ooba_request_timeout', 300),
			self.runner.openai_client
		)

	def parse_and_score(self, inference, ref, ref_fullscale):
		if self.args.l == "de":
			first_pass_answers, revised_answers = parse_answers_de(inference, self.args.revise)
		else:
			first_pass_answers, revised_answers = parse_answers(inference, self.args.revise)
		
		parsed_answers = {
			'first_pass': first_pass_answers,
			'revised': revised_answers
		}

		first_pass_score = calculate_score(ref, first_pass_answers)
		revised_score = calculate_score(ref, revised_answers) if self.args.revise else None

		scores = {
			'first_pass_score': first_pass_score,
			'revised_score': revised_score
		}

		if ref_fullscale:
			first_pass_score_fullscale = calculate_score_fullscale(ref_fullscale, first_pass_answers)
			revised_score_fullscale = calculate_score_fullscale(ref_fullscale, revised_answers) if self.args.revise else None
			scores['first_pass_score_fullscale'] = first_pass_score_fullscale
			scores['revised_score_fullscale'] = revised_score_fullscale

		return scores, parsed_answers

	def store_results(self, question_id, scores, parsed_answers, inference, run_iter):
		iter_results = self.results[self.run_index]['iterations'][str(run_iter)]
		iter_results['respondent_answers'][question_id] = parsed_answers
		iter_results['individual_scores'][question_id] = scores
		iter_results['individual_scores_fullscale'][question_id] = {
			'first_pass_score': scores.get('first_pass_score_fullscale'),
			'revised_score': scores.get('revised_score_fullscale')
		}
		iter_results['raw_inference'][question_id] = inference

	def calculate_score(self):
		# This method would calculate the final score across all iterations
		# For now, we'll leave it as a placeholder
		pass

	def save_results(self):
		safe_dump(self.results, './raw_results.json')
		# Additional logic for saving to database or other formats can be added here
