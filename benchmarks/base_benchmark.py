from abc import ABC, abstractmethod
from typing import Dict, Any
from benchmarks.config_handler import ConfigHandler
import argparse

class BaseBenchmark:
	def __init__(self, config, args, benchmark_config, runner):
		self.config = config
		self.args = args
		self.benchmark_config = benchmark_config
		self.runner = runner
		self.results = {}
		self.run_index = self.generate_run_index()

	def run(self):
		raise NotImplementedError

	def generate_run_index(self):
		raise NotImplementedError