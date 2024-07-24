import json
import datetime
import re
import time
from tqdm import tqdm
from benchmarks.base_benchmark import BaseBenchmark
from lib.run_query import run_query
from lib.scoring import calculate_score, calculate_score_fullscale, parse_answers, parse_answers_de, calculate_eq_bench_score, calculate_item_score_v3
from lib.util import safe_dump, remove_revision_instructions
from lib.run_bench_helper_functions import format_include_exclude_string

class EQBench(BaseBenchmark):
	def __init__(self, config, args, benchmark_config, runner):
		super().__init__(config, args, benchmark_config, runner)
		self.eqbench_version = self.determine_version()
		self.questions = self.load_questions()
		self.total_items = self.calculate_total_items()
		self.progress_bar = None
		self.total_score = 0  # Variable to track the total score
		self.scored_items = 0  
	
	def get_benchmark_type(self):
		return 'eq-bench'

	def determine_version(self):
		if self.args.v1:
			return "v1"
		elif self.args.v3:
			return "v3"
		else:
			return "v2"  # Default to v2 if neither v1 nor v3 is specified

	def update_benchmark_specific_metadata(self, metadata):
		metadata.update({
			"eq_bench_version": self.eqbench_version,
			"language": self.args.l,
			"instruction_template": self.benchmark_config['prompt_type'],
			"model_path": self.benchmark_config['model_path'],
			"lora_path": self.benchmark_config['lora_path'],
			"bitsandbytes_quant": self.benchmark_config['quantization']
		})

	def get_iteration_template(self):
		return {
			'respondent_answers': {},
			'individual_scores': {},
			'individual_scores_fullscale': {},
			'raw_inference': {}
		}
	
	def calculate_total_items(self):
		total = 0
		for question in self.questions.values():
			if self.eqbench_version == "v3":
					total += len(question['reference_answers_conflict_dialogue'])
					#total += len(question['reference_answers_template_questions'])
			else:
					total += 1  # For v1 and v2, each question is one item
		return total * self.benchmark_config['n_iterations']

	def load_questions(self):
		questions_fn = self.get_questions_filename()
		with open(questions_fn, 'r', encoding='utf-8') as f:
			return json.load(f)

	def get_questions_filename(self):
		if self.eqbench_version == "v1":
			return './data/eq_bench_v1_questions_60.json'
		elif self.eqbench_version == "v3":
			return './data/eq_bench_v3_questions.json'
		else:  # v2
			base_filename = './data/eq_bench_v2_questions_171.json'
			if self.args.l != 'en':
					base_name, ext = base_filename.rsplit('.', 1)
					return f"{base_name}_{self.args.l}.{ext}"
			return base_filename

	def generate_run_index(self):
		components = [
			self.benchmark_config['run_id'],
			self.determine_version(),
			self.args.l,
			self.benchmark_config['model_path'],
			self.benchmark_config['lora_path'],
			self.benchmark_config['prompt_type'],
			self.benchmark_config['quantization'],
			self.benchmark_config['inference_engine'],
			self.benchmark_config['ooba_params'],
			format_include_exclude_string(self.benchmark_config['include_patterns'], self.benchmark_config['exclude_patterns'])
		]
		components = [component if component is not None else '' for component in components]
		return "--".join(components)

	def run(self):
		self.progress_bar = tqdm(total=self.total_items, desc="EQ-Bench Progress - Avg Score: N/A", unit="item")
		
		for run_iter in range(1, self.benchmark_config['n_iterations'] + 1):
			print(f"Iteration {run_iter} of {self.benchmark_config['n_iterations']}")
			self.initialize_results()
			
			for question_id, question in self.questions.items():
					if self.is_question_completed(question_id, run_iter):
						self.update_progress_bar(question, completed=True)
						continue
					
					if not self.runner.model and not self.runner.ooba_instance:
						self.runner.initialize_model_or_ooba(self.benchmark_config)
					
					if self.eqbench_version == "v3":
						self.process_v3_question(question_id, question, run_iter)
					else:
						self.process_question_v1_v2(question_id, question, run_iter)
					
					self.save_results()
					if self.eqbench_version != 'v3':
						self.update_progress_bar(question)

		self.progress_bar.close()
		self.print_results()


	def update_progress_bar(self, question, completed=False):
		if self.eqbench_version == "v3":
			if completed:
					#self.progress_bar.update(len(question['reference_answers_conflict_dialogue']) +
					#								len(question['reference_answers_template_questions']))
					self.progress_bar.update(len(question['reference_answers_conflict_dialogue']))
			else:
					self.progress_bar.update(1)  # Update by 1 for each processed item
		else:
			self.progress_bar.update(1)  # For v1 and v2, each question is one item

		# Update the average score in the progress bar description
		if self.scored_items > 0:
			avg_score = self.total_score / self.scored_items
			self.progress_bar.set_description(f"EQ-Bench Progress - Avg Score: {avg_score:.2f}")


	def is_question_completed(self, question_id, run_iter):
		if self.eqbench_version != "v3":
			return (self.run_index in self.results and
					str(run_iter) in self.results[self.run_index]['iterations'] and
					str(question_id) in self.results[self.run_index]['iterations'][str(run_iter)]['individual_scores'])
		
		if (self.run_index not in self.results or
			str(run_iter) not in self.results[self.run_index]['iterations'] or
			question_id not in self.results[self.run_index]['iterations'][str(run_iter)]['individual_scores']):
			return False
		
		scores = self.results[self.run_index]['iterations'][str(run_iter)]['individual_scores'][question_id]
		question = self.questions[question_id]
		
		if False:
			if ('conflict_dialogue' not in scores or
				len(scores['conflict_dialogue']) != len(question['reference_answers_conflict_dialogue']) or
				'template_questions' not in scores or
				len(scores['template_questions']) != len(question['eqbench_template_questions_prompt'])):
				return False
		if ('conflict_dialogue' not in scores or
				len(scores['conflict_dialogue']) != len(question['reference_answers_conflict_dialogue'])):
				return False
		
		return True
  
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
		if self.eqbench_version == "v3":
			self.process_v3_question(question_id, question, run_iter)
		else:
			self.process_question_v1_v2(question_id, question, run_iter)

	def process_v3_question(self, question_id, question, run_iter):
		conflict_dialogue_scores = self.process_v3_conflict_dialogue(question_id, question, run_iter)
		#template_question_scores = self.process_v3_template_questions(question_id, question, run_iter)

		# Update the total score and number of scored items
		for score in conflict_dialogue_scores:
			if score is not None:
					self.total_score += score
					self.scored_items += 1
		#for score in template_question_scores:
		#	if score is not None:
		#			self.total_score += score
		#			self.scored_items += 1

		self.store_v3_scores(question_id, run_iter, {
			'conflict_dialogue': conflict_dialogue_scores,
		#	'template_questions': template_question_scores
		})


	def process_v3_conflict_dialogue(self, question_id, question, run_iter):
		scores = []
		for i, prompt in enumerate(question['eqbench_conflict_dialogue_prompts']):
			inference = self.run_inference(prompt, 0.01)			
			parsed_answers = self.parse_v3_conflict_dialogue_answers(inference)
			reference_answer = question['reference_answers_conflict_dialogue'][i]['reference_answer']
			score = calculate_item_score_v3(reference_answer, parsed_answers) if parsed_answers else None
			if self.args.v:
				print(inference)
				print('Score:', score)
			scores.append(score)
			self.store_v3_results(question_id, parsed_answers, inference, run_iter, 'conflict_dialogue', i)
			self.update_progress_bar(question)
		return scores

	def process_v3_template_questions(self, question_id, question, run_iter):
		scores = []		
		for i, prompt in enumerate(question['eqbench_template_questions_prompt']):
			inference = self.run_inference(prompt, 0.01)
			if inference:
				parsed_answer = self.parse_v3_template_question_answers(inference)
				if parsed_answer and i < len(question['reference_answers_template_questions']):
						reference_answer = question['reference_answers_template_questions'][i]
						score = calculate_item_score_v3(reference_answer['options'], parsed_answer)
				else:
						score = None
			else:
				score = None
			if self.args.v:
				print(inference)
				print('Score:', score)
			scores.append(score)
			self.store_v3_results(question_id, parsed_answer, inference, run_iter, 'template_questions', i)
			self.update_progress_bar(question)
		return scores

	

	def parse_v3_conflict_dialogue_answers(self, inference):
		try:
			# Find the JSON part of the inference
			json_match = re.search(r'\{.*\}', inference, re.DOTALL)
			if json_match:
					json_str = json_match.group(0)
					return json.loads(json_str)
			else:
					print("No JSON found in the inference")
					return None
		except json.JSONDecodeError:
			print("Failed to parse JSON from inference")
			return None

	def parse_v3_template_question_answers(self, inference):
		try:
			# Find the JSON part of the inference
			json_match = re.search(r'\{.*\}', inference, re.DOTALL)
			if json_match:
					json_str = json_match.group(0)
					parsed_answer = json.loads(json_str)
					if all(key in parsed_answer for key in ['option1', 'option2', 'option3', 'option4']):
						return parsed_answer
			print("Failed to parse template question answer")
			return None
		except json.JSONDecodeError:
			print("Failed to parse JSON from inference")
			return None

	def store_v3_results(self, question_id, parsed_answers, inference, run_iter, question_type, index):
		iter_results = self.results[self.run_index]['iterations'][str(run_iter)]
		if question_id not in iter_results['respondent_answers']:
			iter_results['respondent_answers'][question_id] = {}
		if question_type not in iter_results['respondent_answers'][question_id]:
			iter_results['respondent_answers'][question_id][question_type] = []
		iter_results['respondent_answers'][question_id][question_type].append(parsed_answers)
		
		if question_id not in iter_results['raw_inference']:
			iter_results['raw_inference'][question_id] = {}
		if question_type not in iter_results['raw_inference'][question_id]:
			iter_results['raw_inference'][question_id][question_type] = []
		iter_results['raw_inference'][question_id][question_type].append(inference)

	def store_v3_scores(self, question_id, run_iter, scores):
		iter_results = self.results[self.run_index]['iterations'][str(run_iter)]
		if 'individual_scores' not in iter_results:
			iter_results['individual_scores'] = {}
		iter_results['individual_scores'][question_id] = scores

	def calculate_v3_question_score(self, question_id, run_iter):
		conflict_dialogue_scores = []
		template_question_scores = []
		
		respondent_answers = self.results[self.run_index]['iterations'][str(run_iter)]['individual_scores'][question_id]
		
		if 'conflict_dialogue' in respondent_answers:
			conflict_dialogue_scores = [score for score in respondent_answers['conflict_dialogue'] if score is not None]
		
		#if 'template_questions' in respondent_answers:
		#	template_question_scores = [score for score in respondent_answers['template_questions'] if score is not None]
		
		total_scores = conflict_dialogue_scores + template_question_scores
		
		if not total_scores:
			return None
		
		return sum(total_scores) / len(total_scores)


	def calculate_question_score(self, question_id, run_iter):
		if self.eqbench_version == "v3":
			return self.calculate_v3_question_score(question_id, run_iter)
		elif self.eqbench_version == "v2":
			return calculate_score_fullscale(self.questions[question_id]['reference_answer'], self.results[self.run_index]['iterations'][str(run_iter)]['respondent_answers'][question_id])
		else:  # v1
			return calculate_score(self.questions[question_id]['reference_answer'], self.results[self.run_index]['iterations'][str(run_iter)]['respondent_answers'][question_id])


	def process_question_v1_v2(self, question_id, question, run_iter):
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

					# Update the total score and number of scored items
					if scores['first_pass_score'] is not None:
						self.total_score += scores['first_pass_score']
						self.scored_items += 1
					if scores.get('revised_score') is not None:
						self.total_score += scores['revised_score']
						self.scored_items += 1

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
		completion_tokens = 600 if self.args.revise else 60
		if self.eqbench_version == 'v3':
			completion_tokens = 300

		tries = 0
		inference = None
		max_tries = 5
		while not inference and tries < max_tries:
			tries += 1
			inference = run_query(
				self.benchmark_config['model_path'],
				self.benchmark_config['prompt_type'],
				prompt,
				[],  # history
				completion_tokens,  # max_tokens
				self.runner.model,
				self.runner.tokenizer,
				temp,
				self.benchmark_config['inference_engine'],
				self.runner.ooba_instance,
				self.config.get_bool('Oobabooga config', 'automatically_launch_ooba'),
				self.config.get_int('Oobabooga config', 'ooba_request_timeout', 300),
				self.runner.openai_client
			)
			if not inference and tries < max_tries:
				print('Retrying in 10s...')
				time.sleep(10)

		return inference

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

	def print_results(self):
		formatted_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		print(f"----EQ-Bench Benchmark Complete----")
		print(formatted_datetime)
		print('Time taken:', round((datetime.datetime.now() - self.start_time).total_seconds() / 60, 1), 'mins')
		print('Prompt Format:', self.benchmark_config['prompt_type'])
		print('Model:', self.benchmark_config['model_path'])
		if self.benchmark_config['lora_path']:
			print('Lora:', self.benchmark_config['lora_path'])

		lang_suffix = '_' + self.args.l if self.args.l != 'en' else ''
		#score, parseable = calculate_eq_bench_score(self.run_index, self.results, self.RAW_RESULTS_PATH, self.eqbench_version)
		score, parseable = calculate_eq_bench_score(self.run_index, self.results, self.RAW_RESULTS_PATH, self.questions, self.eqbench_version)
		print(f"Score ({self.eqbench_version}{lang_suffix}):", score)
		print('Parseable:', parseable, 'of', self.total_items)

		if parseable / self.total_items < 0.8:
			print("! Benchmark Failed: Less than 80% of questions were parseable")

	def save_results(self):
		safe_dump(self.results, './raw_results.json')
		# Additional logic for saving to database or other formats can be added here
