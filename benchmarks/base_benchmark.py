from abc import ABC, abstractmethod
from typing import Dict, Any
from benchmarks.config_handler import ConfigHandler
import datetime
import os
import json
from lib.util import safe_dump
from lib.run_bench_helper_functions import fix_results


class BaseBenchmark:
	def __init__(self, config, args, benchmark_config, runner):
		self.config = config
		self.args = args
		self.benchmark_config = benchmark_config
		self.runner = runner
		self.RAW_RESULTS_PATH = './raw_results.json'
		self.results = self.load_existing_results()
		self.run_index = self.generate_run_index()
		self.start_time = datetime.datetime.now()		

	def load_existing_results(self):
		if os.path.exists(self.RAW_RESULTS_PATH):
			with open(self.RAW_RESULTS_PATH, 'r') as f:
					return json.load(f)
		return {}

	def generate_run_index(self):
		# Revert to original run index generation
		components = [
			self.benchmark_config['run_id'],
			self.benchmark_config['model_path'],
			self.benchmark_config['lora_path'],
			self.benchmark_config['prompt_type'],
			self.benchmark_config['quantization'],
			self.benchmark_config['inference_engine'],
			self.benchmark_config['ooba_params']
		]
		components = [component if component is not None else '' for component in components]
		return "--".join(components)

	def initialize_results(self):
		results = {}
		if not self.args.w and os.path.exists(self.RAW_RESULTS_PATH):
			with open(self.RAW_RESULTS_PATH, 'r') as f:
					results = json.load(f)
			if self.get_benchmark_type() == 'eq-bench':
					results = fix_results(results)

		if self.run_index not in results:
			results[self.run_index] = {
					'run_metadata': {
						"run_id": self.benchmark_config['run_id'],
						"benchmark_type": self.get_benchmark_type(),
						"total_iterations": self.benchmark_config['n_iterations'],
						"inference_engine": self.benchmark_config['inference_engine'],
						"ooba_params": self.benchmark_config['ooba_params'],
						"include_patterns": self.benchmark_config['include_patterns'],
						"exclude_patterns": self.benchmark_config['exclude_patterns']
					},
					'iterations': {}
			}
			self.update_benchmark_specific_metadata(results[self.run_index]['run_metadata'])

		self.initialize_iterations(results)
		return results

	def initialize_iterations(self, results):
		if 'iterations' not in results[self.run_index]:
			results[self.run_index]['iterations'] = {}
		for run_iter in range(1, self.benchmark_config['n_iterations'] + 1):
			run_iter = str(run_iter)
			if run_iter not in results[self.run_index]['iterations'] or self.args.w:
					results[self.run_index]['iterations'][run_iter] = self.get_iteration_template()

	def get_benchmark_type(self):
		raise NotImplementedError

	def update_benchmark_specific_metadata(self, metadata):
		raise NotImplementedError

	def get_iteration_template(self):
		raise NotImplementedError

	def save_results(self):
		safe_dump(self.results, self.RAW_RESULTS_PATH)

	def print_results(self):
		raise NotImplementedError

	def run(self):
		raise NotImplementedError