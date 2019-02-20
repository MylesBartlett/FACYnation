import configparser


class Settings:
    """Object that holds the configuration"""
    def __init__(self):
        self._key_list = []

    def set_str(self, section, name):
        """Set string configuration value from `section`"""
        self.__set(name, section.get(name))

    def set_int(self, section, name):
        """Set integer configuration value from `section`"""
        try:
            self.__set(name, section.getint(name))
        except ValueError:
            raise ValueError(f"Config file: the value for \"{name}\" has to be of type int.")

    def set_float(self, section, name):
        """Set float configuration value from `section`"""
        try:
            self.__set(name, section.getfloat(name))
        except ValueError:
            raise ValueError(f"Config file: the value for \"{name}\" has to be of type float.")

    def set_bool(self, section, name):
        """Set boolean configuration value from `section`"""
        try:
            self.__set(name, section.getboolean(name))
        except ValueError:
            raise ValueError(f"Config file: the value for \"{name}\" has to be of type boolean.")

    def __set(self, name, value):
        self._key_list.append(name)
        self.__setattr__(name, value)
        if value is None:
            print(f"Warning: no value specified for \"{name}\" (value is therefore set to None).")

    def state_dict(self):
        """Convert this settings object to a dictionary"""
        return {name: getattr(self, name) for name in self._key_list}

    def load_state_dict(self, dictionary):
        """Load settings from a dictionary"""
        for key, value in dictionary.iter():
            self.__setattr__(key, value)


def parse_arguments(config_file):
    """
    This function basically just checks if all config values have the right type.
    """
    args = Settings()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    general_config = get_section(parser, 'general')
    args.set_str(general_config, 'model')

    # MCMC settings
    mcmc_config = get_section(parser, 'mcmc')
    args.set_int(mcmc_config, 'chains')
    args.set_int(mcmc_config, 'iter')
    args.set_bool(mcmc_config, 'verbose')
    args.set_int(mcmc_config, 'seed')

    return args


def get_section(parser, name):
    """Get section of the config parser. Creates section if it doesn't exist"""
    if not parser.has_section(name):
        parser.add_section(name)
    return parser[name]
