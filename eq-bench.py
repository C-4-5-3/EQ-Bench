import argparse
import os
import time
import io
import signal
import sys
import datetime
import csv
import openai
from typing import List, Dict, Any, Tuple
from benchmarks.base_benchmark import BaseBenchmark
from benchmarks.eq_bench import EQBench
from benchmarks.creative_writing_bench import CreativeWritingBench
from benchmarks.judgemark_bench import JudemarkBench
from benchmarks.config_handler import ConfigHandler
from lib.util import parse_batch, preprocess_config_string, revert_placeholders_in_config, is_writing, gpu_cleanup, upload_results_google_sheets
import lib.db
from lib.scoring import calculate_eq_bench_score, calculate_creative_writing_score
from lib.db import save_eq_bench_result_to_db, save_creative_writing_result_to_db, save_judgemark_result_to_db
from lib.util import gpu_cleanup, delete_symlinks_and_dir
from lib.run_bench_helper_functions import format_include_exclude_string
from lib.load_model import load_model
import lib.ooba

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

BENCH_RESULTS_PATH = './benchmark_results.csv'
DEBUG=True

class BenchmarkRunner:
	def __init__(self, config, args):
		self.config = config
		self.args = args
		self.model = None
		self.tokenizer = None
		self.ooba_instance = None
		self.models_to_delete = {}
		self.models_remaining = []
		self.openai_client = None

	def run(self):
		self.setup_environment()
		parsed_batch = self.prepare_batch()
		self.run_benchmarks(parsed_batch)
  
	def setup_openai_client(self, str_to_replace='', replace_with=''):
		api_key = self.config.get('OpenAI', 'api_key', '')
		base_url = 'https://api.openai.com/v1/'		
		alt_url = self.config.get('OpenAI', 'openai_compatible_url', '')		
		
		if alt_url:
			if str_to_replace:
				alt_url = alt_url.replace(str_to_replace, replace_with)
			base_url = alt_url
			print('base_url', base_url)

		

		if api_key:
			return openai.OpenAI(
					api_key=api_key,
					base_url=base_url
			)
		return None

	def setup_environment(self):
		# Set up environment variables and configurations
		os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1' if self.args.f else '0'
		if self.config.get('Huggingface', 'access_token'):
			os.environ["HF_TOKEN"] = self.config.get('Huggingface', 'access_token')
			from huggingface_hub import login
			login(token=self.config.get('Huggingface', 'access_token'))
		if os.path.exists('./firebase_creds.json'):
			os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath('./firebase_creds.json')
			lib.db.init_db()

	def prepare_batch(self) -> List[Dict[str, Any]]:
		preprocessed_benchmark_runs = revert_placeholders_in_config(self.config['Benchmarks to run'])
		return parse_batch(preprocessed_benchmark_runs, 
								self.config.get('Oobabooga config', 'ooba_launch_script'),
								self.config.get_bool('Oobabooga config', 'automatically_launch_ooba'))

	def run_benchmarks(self, parsed_batch: List[Tuple]):
		start_time = time.time()
		for i, batch_item in enumerate(parsed_batch, 1):
			(run_id, prompt_type, model_path, lora_path, quantization, 
				n_iterations, inference_engine, ooba_params, include_patterns, exclude_patterns) = batch_item

			print(f'--------------\nRunning benchmark {i} of {len(parsed_batch)}\n')
			print(model_path)
			if lora_path:
					print(lora_path)
			print('--------------')

			# This is here because huggingface pro api uses the model in the api url
			openai_compatible_url = self.config.get('OpenAI', 'openai_compatible_url', '')
			if inference_engine == 'openai' and openai_compatible_url and 'https://api-inference.huggingface.co' in openai_compatible_url:
				self.openai_client = None
				self.openai_client = self.setup_openai_client(str_to_replace='<MODEL>', replace_with=model_path)
			if not self.openai_client:
				self.openai_client = self.setup_openai_client()

			benchmark_config = {
					'run_id': run_id,
					'prompt_type': prompt_type,
					'model_path': model_path,
					'lora_path': lora_path,
					'quantization': quantization,
					'n_iterations': n_iterations,
					'inference_engine': inference_engine,
					'ooba_params': ooba_params,
					'include_patterns': include_patterns,
					'exclude_patterns': exclude_patterns
			}

			self.prepare_model_deletion(benchmark_config)

			if DEBUG:
				# Debug mode: let errors propagate
				for benchmark_type in self.args.benchmarks:
						benchmark = self.create_benchmark(benchmark_type, benchmark_config)
						benchmark.run()
						self.save_and_upload_results(benchmark, benchmark_type, benchmark_config)
				self.cleanup(benchmark_config)
			else:
					# Normal mode: use try-except blocks
					try:
						for benchmark_type in self.args.benchmarks:
							benchmark = self.create_benchmark(benchmark_type, benchmark_config)
							benchmark.run()
							self.save_and_upload_results(benchmark, benchmark_type, benchmark_config)
					except KeyboardInterrupt:
						self.cleanup(benchmark_config)
						raise
					except Exception as e:
						print(e)
						self.cleanup(benchmark_config)
					finally:
						self.cleanup(benchmark_config)

			self.cleanup(benchmark_config)
			self.models_remaining = self.models_remaining[1:]

		end_time = time.time()
		print('---------------')
		print('Batch completed')
		print('Time taken:', round((end_time - start_time) / 60, 1), 'mins')
		print('---------------')

	def save_and_upload_results(self, benchmark, benchmark_type, benchmark_config):
		formatted_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		run_id = benchmark_config['run_id']
		model_path = benchmark_config['model_path']
		lora_path = benchmark_config['lora_path']
		prompt_type = benchmark_config['prompt_type']
		quantization = benchmark_config['quantization']
		inference_engine = benchmark_config['inference_engine']
		ooba_params = benchmark_config['ooba_params']
		include_patterns = benchmark_config['include_patterns']
		exclude_patterns = benchmark_config['exclude_patterns']

		# Calculate score based on benchmark type
		if benchmark_type == 'eq-bench':
			benchmark_version_short = f"{'v3' if self.args.v3 else 'v1' if self.args.v1 else 'v2'}{self.args.l if self.args.l != 'en' else ''}"
			benchmark_version = f"{benchmark_type}_{benchmark_version_short}"
			#score, parseable = calculate_eq_bench_score(benchmark.run_index, benchmark.results, './raw_results.json', benchmark_version_short)
			score, parseable = calculate_eq_bench_score(benchmark.run_index, benchmark.results, './raw_results.json', benchmark.questions, benchmark_version_short)
			

		elif benchmark_type == 'creative-writing':
			score = calculate_creative_writing_score(benchmark.run_index, benchmark.results, './raw_results.json')
			parseable = 'N/A'
			benchmark_version = 'creative-writing'
		elif benchmark_type == 'judgemark':
			score = benchmark.results[benchmark.run_index]['judgemark_results']['mean_score']
			parseable = 'N/A'
			benchmark_version = 'judgemark'

		# Prepare result data
		result_data = [
			run_id,
			formatted_datetime,
			prompt_type,
			model_path,
			lora_path,
			quantization,
			round(score, 2) if score is not None else 'FAILED',
			benchmark_version,
			parseable,
			benchmark.benchmark_config['n_iterations'],
			inference_engine,
			ooba_params,
			format_include_exclude_string(include_patterns, exclude_patterns),
			''  # Error field, empty if successful
		]

		# Save to CSV
		self.save_to_csv(result_data)

		# Upload to Google Sheets
		google_spreadsheet_url = self.config.get('Results upload', 'google_spreadsheet_url')
		if google_spreadsheet_url and os.path.exists('./google_creds.json'):
			upload_results_google_sheets(google_spreadsheet_url, result_data)

		# Save to database
		self.save_to_database(benchmark, benchmark_type, score, parseable)

	def prepare_model_deletion(self, benchmark_config):
		model_path = benchmark_config['model_path']
		include_patterns = benchmark_config['include_patterns']
		exclude_patterns = benchmark_config['exclude_patterns']
		if model_path and not os.path.exists(model_path):
			this_model_key = model_path + '_' + ','.join(include_patterns) + '_' + ','.join(exclude_patterns)
			if self.args.d:
					self.models_to_delete[this_model_key] = True
			self.models_remaining.append(this_model_key)

	def save_to_csv(self, result_data):
		file_exists = os.path.isfile(BENCH_RESULTS_PATH)
		
		with open(BENCH_RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			if not file_exists:
				writer.writerow([
						'Run ID', 'Timestamp', 'Prompt Format', 'Model Path', 'Lora Path', 
						'Quantization', 'Benchmark Score', 'Benchmark Version', 
						'Num Questions Parseable', 'Num Iterations', 'Inference Engine', 
						'Ooba Params', 'Download Filters', 'Error'
				])
			writer.writerow(result_data)

	def save_to_database(self, benchmark, benchmark_type, score, parseable):
		if benchmark_type == 'eq-bench':
			save_eq_bench_result_to_db(benchmark.results[benchmark.run_index], score, parseable, '', benchmark.run_index, True)
		elif benchmark_type == 'creative-writing':
			save_creative_writing_result_to_db(benchmark.results[benchmark.run_index], score, 'N/A', '', benchmark.run_index, True)
		elif benchmark_type == 'judgemark':
			save_judgemark_result_to_db(benchmark.results[benchmark.run_index], score, 'N/A', '', benchmark.run_index, True)
   
	def create_benchmark(self, benchmark_type: str, benchmark_config: Dict[str, Any]) -> BaseBenchmark:
		if benchmark_type == 'eq-bench':
			return EQBench(self.config, self.args, benchmark_config, self)
		elif benchmark_type == 'creative-writing':
			return CreativeWritingBench(self.config, self.args, benchmark_config, self)
		elif benchmark_type == 'judgemark':
			return JudemarkBench(self.config, self.args, benchmark_config, self)
		else:
			raise ValueError(f"Invalid benchmark type: {benchmark_type}")

	def initialize_model_or_ooba(self, benchmark_config):
		inference_engine = benchmark_config['inference_engine']
		model_path = benchmark_config['model_path']
		lora_path = benchmark_config['lora_path']
		quantization = benchmark_config['quantization']

		if inference_engine == 'transformers' and self.model is None:
			self.model, self.tokenizer = load_model(model_path, lora_path, quantization, trust_remote_code=self.config.get_bool('Options', 'trust_remote_code', False))
		elif inference_engine == 'ooba' and self.ooba_instance is None:
			ooba_launch_script = self.config.get('Oobabooga config', 'ooba_launch_script')
			if not ooba_launch_script:
					raise ValueError("ooba_launch_script not set in config")
			
			self.ooba_instance = lib.ooba.Ooba(
					ooba_launch_script, model_path, 
					self.config.get('Huggingface', 'cache_dir'), 
					self.args.v,
					trust_remote_code=self.config.get_bool('Options', 'trust_remote_code', False),
					ooba_args_global=self.config.get('Oobabooga config', 'ooba_params_global', ''),
					ooba_args=benchmark_config['ooba_params'],
					fast_download=self.args.f,
					include_patterns=benchmark_config['include_patterns'],
					exclude_patterns=benchmark_config['exclude_patterns'],
					hf_access_token=self.config.get('Huggingface', 'access_token')
			)
			ooba_started_ok = self.ooba_instance.start()
			if not ooba_started_ok:
					raise Exception("Ooba failed to launch.")

	def cleanup(self, benchmark_config):
		inference_engine = benchmark_config['inference_engine']
		model_path = benchmark_config['model_path']
		include_patterns = benchmark_config['include_patterns']
		exclude_patterns = benchmark_config['exclude_patterns']

		require_gpu_cleanup = False
		if self.model:
			require_gpu_cleanup = True
			del self.model
			self.model = None
		if self.tokenizer:
			del self.tokenizer
			self.tokenizer = None
		if inference_engine == 'ooba' and self.ooba_instance:
			require_gpu_cleanup = True
			try:
					self.ooba_instance.stop()
			except Exception as e:
					pass
			self.ooba_instance = None

		if self.args.d and self.models_to_delete:
			this_model_key = model_path + '_' + ','.join(include_patterns) + '_' + ','.join(exclude_patterns)
			if model_path and this_model_key in self.models_to_delete and this_model_key not in self.models_remaining[1:]:
					if inference_engine == 'transformers':
						dir_to_delete = os.path.expanduser('~/.cache/huggingface/hub/models--' + model_path.replace('/', '--').replace('\\', '--'))
						if os.path.exists(dir_to_delete):
							delete_symlinks_and_dir(dir_to_delete, self.args.v)
						else:
							print('! Cache not found:', dir_to_delete)
					elif inference_engine == 'ooba':
						if self.ooba_instance and self.ooba_instance.model_downloaded_fullpath:
							dir_to_delete = self.ooba_instance.model_downloaded_fullpath
							if os.path.exists(dir_to_delete):
									delete_symlinks_and_dir(dir_to_delete, self.args.v)
							else:
									print('! Cache not found:', dir_to_delete)
		if require_gpu_cleanup:
			gpu_cleanup()

	def final_cleanup(self):
		if self.ooba_instance:
			print('Stopping ooba...')
			self.ooba_instance.stop()
		time.sleep(2)
		self.model = None
		self.tokenizer = None
		self.ooba_instance = None
		#gpu_cleanup()
		while is_writing:
			print('Waiting for writes to complete...')
			time.sleep(0.1)
		print('\nAll writes completed. Exiting...')

def main():
	parser = argparse.ArgumentParser(description="Run benchmark pipeline based on specified configuration.")
	parser.add_argument('-v1', action='store_true', help="Run v1 of EQ-Bench (legacy).")
	parser.add_argument('-v3', action='store_true', help="Run v3 of EQ-Bench (new format).")
	parser.add_argument('-revise', action='store_true', help="Include the revision component of the test.")
	parser.add_argument('--benchmarks', nargs='+', default=['eq-bench'],
							help="Specify the benchmark types to run.")
	parser.add_argument('-w', action='store_true', help="Overwrites existing results.")
	parser.add_argument('-d', action='store_true', help="Delete downloaded models after benchmark.")
	parser.add_argument('-f', action='store_true', help="Use hftransfer for downloading models.")
	parser.add_argument('-v', action='store_true', help="Display more verbose output.")
	parser.add_argument('-l', default='en', help="Set the language of the question dataset.")
	parser.add_argument('-r', type=int, default=5, help="Set the number of retries for failed benchmark runs.")
	args = parser.parse_args()

	config_content = preprocess_config_string('config.cfg')
	config_file_iostream = io.StringIO(config_content)
	config = ConfigHandler(config_file_iostream)

	runner = BenchmarkRunner(config, args)

	def signal_handler(sig, frame):
		runner.final_cleanup()
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)

	runner.run()

if __name__ == '__main__':
	main()