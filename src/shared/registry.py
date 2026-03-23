import logging

# Set up basic logging for the registry
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Registry:
    """
    A simple registry to map string names to classes.
    Allows for decoupled, config-driven instantiation of modules.
    """
    def __init__(self, name: str):
        self._name = name
        self._module_dict = dict()

    def register(self, name: str = None):
        """Decorator to register a class with the registry."""
        def _register(cls):
            module_name = name if name else cls.__name__
            if module_name in self._module_dict:
                logger.warning(f"⚠️ Overwriting existing module '{module_name}' in {self._name} registry!")
            
            self._module_dict[module_name] = cls
            logger.debug(f"🔌 Registered '{module_name}' into {self._name}")
            return cls
        return _register

    def get(self, name: str):
        """Retrieve a class from the registry by name."""
        if name not in self._module_dict:
            available = list(self._module_dict.keys())
            raise KeyError(f"❌ '{name}' not found in {self._name} registry. Available: {available}")
        return self._module_dict[name]

# Global access points for our three main concerns
PREPROCESSORS = Registry('Preprocessors')
TRAINERS = Registry('Trainers')
HOOKS = Registry('Hooks')
