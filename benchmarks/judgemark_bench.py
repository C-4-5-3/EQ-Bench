import json
import datetime
import tqdm
from benchmarks.base_benchmark import BaseBenchmark
from lib.util import safe_dump
from lib.judgemark import compute_judgemark_results
from lib.creative_writing_utils_v2 import process_writing_prompt

class JudemarkBench(BaseBenchmark):
	def __init__(self, config, args, benchmark_config, runner):
		super().__init__(config, args, benchmark_config, runner)
		self.test_model_outputs = self.load_test_model_outputs()
		self.prompts = self.load_prompts()
		self.total_prompts = sum(len(model_outputs) for model_outputs in self.test_model_outputs.values()) * self.benchmark_config['n_iterations']
		self.progress_bar = None

	def get_benchmark_type(self):
		return 'judgemark'

	def update_benchmark_specific_metadata(self, metadata):
		metadata.update({
			'judge_model': self.config.get('Creative Writing Benchmark', 'judge_model')
		})

	def get_iteration_template(self):
		return {
			'judgemark_results': {}
		}

	def load_test_model_outputs(self):
		with open('data/judgemark_test_set.json', 'r', encoding='utf-8') as f:
			return json.load(f)

	def load_prompts(self):
		with open('data/creative_writing_prompts.json', 'r', encoding='utf-8') as f:
			return json.load(f)

	def generate_run_index(self):
		return f"{self.benchmark_config['run_id']}--judgemark--{self.get_judge_params()['judge_model']}"

	def run(self):
		self.progress_bar = tqdm(total=self.total_prompts, desc="Judgemark Progress", unit="prompt")
		
		for run_iter in range(1, self.benchmark_config['n_iterations'] + 1):
			print(f"Iteration {run_iter} of {self.benchmark_config['n_iterations']}")
			self.initialize_results()
			
			for model_name, model_outputs in self.test_model_outputs.items():
					print(f'########################\nTest model: {model_name}\n########################')
					self.process_model_outputs(model_name, model_outputs, run_iter)

		self.progress_bar.close()
		self.calculate_score()
		self.save_results()
		self.print_results()

	def create_run_metadata(self):
		return {
			"run_id": self.benchmark_config['run_id'],
			"benchmark_type": "judgemark",
			"total_iterations": self.benchmark_config['n_iterations'],
			"judge_model": self.get_judge_params()['judge_model']
		}

	def process_model_outputs(self, model_name, model_outputs, run_iter):
		iter_results = self.results[self.run_index]['iterations'][str(run_iter)]
		if model_name not in iter_results['judgemark_results']:
			iter_results['judgemark_results'][model_name] = {
					'individual_scores': {},
					'test_model_response': {},
					'judge_model_response': {}
			}
		
		for prompt_id, test_model_response in model_outputs.items():
			if self.is_prompt_completed(model_name, prompt_id, run_iter):
					if self.args.v:
						print(f'Prompt {prompt_id} already completed')
					self.progress_bar.update(1)
					continue
			
			prompt_data = self.prompts[prompt_id]
			scores = process_writing_prompt(
					prompt_id, prompt_data, None, None, None, None, self.results,
					self.run_index, str(run_iter), self.args.v, 0,
					self.benchmark_config['inference_engine'], None, False, 300,
					self.runner.openai_client, self.get_judge_params(), test_model_response, model_name
			)
			
			if scores:
					safe_dump(self.results, './raw_results.json')
			
			self.progress_bar.update(1)

	def is_prompt_completed(self, model_name, prompt_id, run_iter):
		return (self.run_index in self.results and
					str(run_iter) in self.results[self.run_index]['iterations'] and
					model_name in self.results[self.run_index]['iterations'][str(run_iter)]['judgemark_results'] and
					prompt_id in self.results[self.run_index]['iterations'][str(run_iter)]['judgemark_results'][model_name]['individual_scores'])

	def get_judge_params(self):
		return {
			'judge_model_api': self.config.get('Creative Writing Benchmark', 'judge_model_api'),
			'judge_model': self.config.get('Creative Writing Benchmark', 'judge_model'),
			'judge_model_api_key': self.config.get('Creative Writing Benchmark', 'judge_model_api_key')
		}

	def calculate_score(self):
		compute_judgemark_results(self.results, self.run_index, self.test_model_outputs, self.args.v)
  
	def print_results(self):
		formatted_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		print(f"----Judgemark Benchmark Complete----")
		print(formatted_datetime)
		print('Time taken:', round((datetime.datetime.now() - self.start_time).total_seconds() / 60, 1), 'mins')
		print('Judge:', self.config.get('Creative Writing Benchmark', 'judge_model'))
		print("Final Judgemark Benchmark Results:")
		print('Mean Score:', round(self.results[self.run_index]['judgemark_results']['mean_score'], 2))
		print('Std. Dev.:', round(self.results[self.run_index]['judgemark_results']['std_dev'], 2))
		print('Judgemark Score:', round(self.results[self.run_index]['judgemark_results']['extended_metrics']['Judgemark'], 2))


	def save_results(self):
		safe_dump(self.results, './raw_results.json')