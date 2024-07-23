import configparser
from typing import Any

class ConfigHandler:
	def __init__(self, config_file):
		self.config = configparser.RawConfigParser(allow_no_value=True)
		self.config.optionxform = str
		self.config.read_file(config_file)

	def get(self, section: str, option: str, fallback: Any = None) -> Any:
		return self.config.get(section, option, fallback=fallback)

	def get_bool(self, section: str, option: str, fallback: bool = False) -> bool:
		value = self.config.get(section, option, fallback=str(fallback))
		return value.lower() in ('yes', 'true', 't', 'y', '1')

	def get_int(self, section: str, option: str, fallback: int = 0) -> int:
		return self.config.getint(section, option, fallback=fallback)

	def __getitem__(self, section: str):
		return dict(self.config[section])